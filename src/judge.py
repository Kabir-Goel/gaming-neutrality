"""Code each collected response against prompts/judge_rubric.md.

The judge sees the response text and nothing else — no prompt, no persona, no
frame. That blinding is the point: stance and framing have to be read off the
text itself, not inferred from the condition that produced it. Responses that
refer to their own context ("your roommate") still leak that much, which is a
property of the corpus rather than something to strip here.

Resumable on the same terms as collect.py: ids already coded are skipped, and
rows are pinned to both the rendered judge prompt and the judge's decoding
config, so editing the rubric or the effort setting invalidates affected codes
instead of leaving them silently mixed with fresh ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from . import models
from .collect import (
    MAX_PER_PROVIDER,
    append_jsonl,
    config_hash,
    quarantine_stale,
    read_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent
RESPONSES_PATH = ROOT / "data" / "raw" / "responses.jsonl"
CODED_PATH = ROOT / "data" / "coded" / "coded.jsonl"
FAILURES_PATH = ROOT / "data" / "coded" / "judge_failures.jsonl"

# Thinking and visible output share this budget, so at effort=high a long
# rationale can be starved: the first full run truncated 3 of 1920 rows
# mid-JSON. 800 leaves headroom; typical replies are ~300 chars.
JUDGE_MAX_TOKENS = 800

# Appended verbatim on the one retry a malformed reply gets.
RETRY_NUDGE = "\n\nOutput valid JSON only."

REQUIRED_KEYS = ("stance", "framing", "refusal", "rationale")

_write_lock = threading.Lock()
_provider_slots: dict[str, threading.Semaphore] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_fences(text: str) -> str:
    """Remove a markdown code fence around a JSON body, if present.

    Nothing guarantees the model honours "output ONLY valid JSON", so this
    tolerates ```json ... ``` and bare ``` ... ``` wrappers.
    """
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    body = cleaned[3:]
    if body[:4].lower().startswith("json"):
        body = body[4:]
    closing = body.rfind("```")
    if closing != -1:
        body = body[:closing]
    return body.strip()


def parse_scores(text: str) -> dict[str, Any] | None:
    """Return the coded scores, or None if the reply is not usable.

    A reply that parses as JSON but lacks a required key is treated as a
    parse failure: a row missing `stance` is not partially useful, and
    silently writing null for it would hide the problem from the analysis.
    """
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or any(key not in data for key in REQUIRED_KEYS):
        return None
    for key in ("stance", "framing"):
        if isinstance(data[key], bool) or not isinstance(data[key], (int, float)):
            return None
    if isinstance(data["refusal"], bool) or not isinstance(data["refusal"], int):
        return None
    if not isinstance(data["rationale"], str):
        return None
    return {
        "stance": float(data["stance"]),
        "framing": float(data["framing"]),
        "refusal": int(data["refusal"]),
        "rationale": data["rationale"],
    }


def build_prompt(rubric: str, issues: dict[str, Any], row: dict[str, Any]) -> str:
    """Render the rubric for this row's issue and attach the response text."""
    return (
        rubric.replace("{issue}", row["issue"]).replace(
            "{positive_direction}", issues[row["issue"]]["positive_direction"]
        )
        + '\n\nRESPONSE TO SCORE:\n"""\n'
        + row["response"]
        + '\n"""\n'
    )


def selected(item: dict[str, Any], args: argparse.Namespace) -> bool:
    """True if a response passes every active filter."""
    if args.model_ids is not None and item["model_id"] not in args.model_ids:
        return False
    if args.issue is not None and item["issue"] != args.issue:
        return False
    if args.persona is not None and item["persona"] != args.persona:
        return False
    if args.frame is not None and item["frame"] != args.frame:
        return False
    return True


def judge_one(
    item: dict[str, Any],
    judge: dict[str, Any],
    decoding: dict[str, Any],
    cfg_hash: str,
    coded_path: Path,
    failures_path: Path,
) -> bool:
    """Code one response and append the result. Returns True on success.

    Success means the call completed, not that the reply parsed — an
    unparseable reply is recorded with parse_ok false so it stays visible in
    the dataset rather than being retried forever.
    """
    provider, name = judge["provider"], judge["name"]
    prompt = item["prompt"]

    with _provider_slots[provider]:
        try:
            result = models.generate(
                provider=provider,
                model=name,
                prompt=prompt,
                temperature=decoding["temperature"],
                top_p=decoding["top_p"],
                max_tokens=JUDGE_MAX_TOKENS,
                # Paired with the config_hash call in main(); both must use
                # this same constant or the row misdescribes its own run.
                anthropic_effort=models.JUDGE_EFFORT,
            )
            scores = parse_scores(result["text"])
            if scores is None:
                result = models.generate(
                    provider=provider,
                    model=name,
                    prompt=prompt + RETRY_NUDGE,
                    temperature=decoding["temperature"],
                    top_p=decoding["top_p"],
                    max_tokens=JUDGE_MAX_TOKENS,
                    anthropic_effort=models.JUDGE_EFFORT,
                )
                scores = parse_scores(result["text"])
        except Exception as exc:  # noqa: BLE001 - a dead call must not stop the run
            append_jsonl(
                failures_path,
                {
                    "id": item["id"],
                    "provider": provider,
                    "judge": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "ts": _now(),
                },
            )
            return False

    append_jsonl(
        coded_path,
        {
            "id": item["id"],
            "stance": scores["stance"] if scores else None,
            "framing": scores["framing"] if scores else None,
            "refusal": scores["refusal"] if scores else None,
            "rationale": scores["rationale"] if scores else None,
            "judge": name,
            "parse_ok": scores is not None,
            "config_hash": cfg_hash,
            # Hash of the rendered judge prompt: rubric text plus the response
            # under judgement, so a rubric edit or a recollected response
            # invalidates the codes that depended on it.
            "prompt_hash": item["prompt_hash"],
            # Kept so a malformed reply can be diagnosed, and so the codes can
            # be re-derived without paying for the calls again.
            "raw": result["text"],
            "finish_reason": result["finish_reason"],
            "latency_ms": result["latency_ms"],
            "ts": _now(),
        },
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="stop after N calls (smoke test)")
    parser.add_argument(
        "--model", help="restrict to one model id or a comma-separated list, e.g. M1,M2"
    )
    parser.add_argument("--issue", help="restrict to one issue, e.g. guns")
    # The judge is blind to persona and frame, but selecting on them is how a
    # balanced sample gets drawn: framing variance lives in those contrasts,
    # and file order alone yields an almost entirely neutral/audit sample.
    parser.add_argument("--persona", help="restrict to one persona, e.g. C")
    parser.add_argument("--frame", help="restrict to one frame, e.g. A")
    # config.yaml's judge is frozen once the main run starts (build guide
    # §4.1), so a second judge for inter-judge reliability has to be an
    # override here rather than an edit to the frozen file. Output path
    # moves with it so a second-judge run can never land in coded.jsonl.
    parser.add_argument(
        "--judge-provider", help="override config.judge.provider, e.g. anthropic"
    )
    parser.add_argument(
        "--judge-model", help="override config.judge.name, e.g. claude-haiku-4-5-20251001"
    )
    parser.add_argument(
        "--output", help="override the coded-output path (default data/coded/coded.jsonl)"
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    issues = yaml.safe_load((ROOT / "prompts" / "issues.yaml").read_text())
    rubric = (ROOT / "prompts" / "judge_rubric.md").read_text()
    judge = dict(config["judge"])
    if args.judge_provider:
        judge["provider"] = args.judge_provider
    if args.judge_model:
        judge["name"] = args.judge_model
    by_id = {model["id"]: model for model in config["models"]}

    coded_path = Path(args.output).resolve() if args.output else CODED_PATH
    failures_path = (
        coded_path.parent / f"{coded_path.stem}_failures.jsonl"
        if args.output
        else FAILURES_PATH
    )

    args.model_ids = None
    if args.model is not None:
        args.model_ids = list(
            dict.fromkeys(part.strip() for part in args.model.split(",") if part.strip())
        )
        if not args.model_ids:
            parser.error("--model given but no model ids found in it")
        unknown = [model for model in args.model_ids if model not in by_id]
        if unknown:
            parser.error(
                f"unknown model(s) {', '.join(unknown)}; "
                f"expected one of {', '.join(by_id)}"
            )
    if args.issue and args.issue not in config["design"]["issues"]:
        parser.error(
            f"unknown issue {args.issue!r}; expected one of "
            f"{', '.join(config['design']['issues'])}"
        )
    if args.persona and args.persona not in config["design"]["personas"]:
        parser.error(
            f"unknown persona {args.persona!r}; expected one of "
            f"{', '.join(config['design']['personas'])}"
        )
    if args.frame and args.frame not in config["design"]["frames"]:
        parser.error(
            f"unknown frame {args.frame!r}; expected one of "
            f"{', '.join(config['design']['frames'])}"
        )

    responses = read_jsonl(RESPONSES_PATH)
    if not responses:
        parser.error(
            f"{RESPONSES_PATH.relative_to(ROOT)} is empty; run src/collect.py"
        )

    # Render every prompt up front so the hashes exist before the staleness
    # check runs, the same way the grid does for collect.py.
    work = []
    for row in responses:
        prompt = build_prompt(rubric, issues, row)
        work.append(
            {
                "id": row["id"],
                "model_id": row["model_id"],
                "issue": row["issue"],
                "persona": row["persona"],
                "frame": row["frame"],
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )

    decoding = dict(config["decoding"], max_tokens=JUDGE_MAX_TOKENS)
    # Paired with the generate() call in judge_one(): same constant, so the
    # recorded hash describes the effort the calls actually ran at.
    judge_hash = config_hash(
        judge["provider"],
        judge["name"],
        decoding,
        anthropic_effort=models.JUDGE_EFFORT,
    )
    expected_config = {model_id: judge_hash for model_id in by_id}

    coded_path.parent.mkdir(parents=True, exist_ok=True)
    coded = quarantine_stale(
        read_jsonl(coded_path),
        {item["id"]: item["prompt_hash"] for item in work},
        # Coded rows carry no model_id: every row is judged by the same judge,
        # so they all map to the one hash.
        {**expected_config, None: judge_hash},
        coded_path,
        label="coded row",
    )
    done = {record["id"] for record in coded}

    pending = [item for item in work if item["id"] not in done and selected(item, args)]
    if args.limit is not None:
        pending = pending[: args.limit]

    _provider_slots.setdefault(
        judge["provider"], threading.Semaphore(MAX_PER_PROVIDER)
    )

    active = [
        text
        for text in (
            ",".join(args.model_ids) if args.model_ids else None,
            args.issue,
            args.persona,
            args.frame,
            f"limit {args.limit}" if args.limit is not None else None,
        )
        if text
    ]
    scope = f" [{', '.join(active)}]" if active else ""
    print(
        f"{len(work)} responses, {len(done)} already coded; "
        f"{len(pending)} to judge{scope}\n"
        f"judge: {judge['provider']}/{judge['name']} "
        f"effort={models.JUDGE_EFFORT} max_tokens={JUDGE_MAX_TOKENS} "
        f"({MAX_PER_PROVIDER} concurrent)\n"
        f"output: {coded_path.relative_to(ROOT) if coded_path.is_relative_to(ROOT) else coded_path}"
    )
    if not pending:
        return 0

    failures = 0
    with ThreadPoolExecutor(max_workers=MAX_PER_PROVIDER) as pool:
        futures = {
            pool.submit(judge_one, item, judge, decoding, judge_hash, coded_path, failures_path): item
            for item in pending
        }
        progress = tqdm(as_completed(futures), total=len(futures), unit="call")
        try:
            for future in progress:
                if not future.result():
                    failures += 1
                    progress.set_postfix(failed=failures)
        except KeyboardInterrupt:
            print("\ninterrupted; completed codes are saved, rerun to resume")
            pool.shutdown(wait=False, cancel_futures=True)
            return 130

    print(f"coded {len(pending) - failures}/{len(pending)}")
    if failures:
        print(
            f"{failures} failed -> {failures_path.relative_to(ROOT)} "
            f"(not marked done; rerun to retry)"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
