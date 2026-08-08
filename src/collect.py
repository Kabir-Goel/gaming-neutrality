"""Execute the call grid and append one response record per API call.

Resumable by construction: ids already in data/raw/responses.jsonl are
skipped, and every result is appended and flushed the moment it arrives, so
an interrupted run loses at most the calls still in flight. Failures are
recorded separately and are *not* treated as done — rerunning retries them.

Concurrency is capped per provider rather than globally, so a slow or
rate-limited provider cannot starve the others.
"""

from __future__ import annotations

import argparse
import collections
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

ROOT = Path(__file__).resolve().parent.parent
GRID_PATH = ROOT / "data" / "raw" / "grid.jsonl"
RESPONSES_PATH = ROOT / "data" / "raw" / "responses.jsonl"
FAILURES_PATH = ROOT / "data" / "raw" / "failures.jsonl"
STALE_PATH = ROOT / "data" / "raw" / "stale.jsonl"

MAX_PER_PROVIDER = 4

_write_lock = threading.Lock()
_provider_slots: dict[str, threading.Semaphore] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, tolerating a truncated final line.

    A run killed mid-write can leave a partial line; that record is simply
    absent from the done set and will be recollected.
    """
    if not path.exists():
        return []
    records = []
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"[collect] skipping unparseable {path.name} line {lineno}",
                    file=sys.stderr,
                )
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record and flush, under a lock shared by all workers."""
    with _write_lock:
        with path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def config_hash(provider: str, name: str, decoding: dict[str, Any]) -> str:
    """Hash the model and decoding settings this run will send to `provider`.

    Hashes the *requested* configuration rather than what a provider turns
    out to accept, because the check has to run before any call has been made
    — effective values are only known after a model answers once. That is
    also the more useful comparison: a parameter the endpoint refuses is
    provider behaviour, recorded per row in effective_*, whereas a change
    here is a deliberate edit to the experiment.

    The configured model name is included because rows are keyed by model_id
    (M1..M5), which is a stable label over a swappable model — repointing M1
    at a different model would otherwise leave its old rows looking current.
    Provider-specific controls are included only where they apply, so adding
    a knob for one provider does not invalidate every other model's rows.
    """
    fields: dict[str, Any] = {
        "model": name,
        "provider": provider,
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "max_tokens": decoding["max_tokens"],
    }
    if provider == "anthropic":
        fields["effort"] = models.ANTHROPIC_EFFORT
    elif provider == "google":
        fields["thinking_level"] = models.GOOGLE_THINKING_LEVEL
    if models.disables_reasoning(provider, name):
        fields["reasoning_enabled"] = False
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def quarantine_stale(
    grid: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    expected_config: dict[str, str],
) -> list[dict[str, Any]]:
    """Move responses that no longer match the current run into stale.jsonl.

    Two ways a row goes stale without its id changing. Editing prompts/*.yaml
    changes prompt text, and editing decoding settings changes the sampling
    condition; either way the row stays in the done set forever and silently
    contaminates the dataset. Comparing stored prompt_hash and config_hash
    against the current run catches both. Quarantined rows are appended to
    stale.jsonl rather than deleted, and dropping them from the done set is
    what causes this run to recollect them.

    Returns the surviving responses.
    """
    expected_prompt = {item["id"]: item["prompt_hash"] for item in grid}
    fresh: list[dict[str, Any]] = []
    stale: list[tuple[dict[str, Any], str | None, str | None, list[str]]] = []

    for record in responses:
        prompt_want = expected_prompt.get(record["id"])
        config_want = expected_config.get(record.get("model_id"))
        reasons: list[str] = []

        if prompt_want is None:
            reasons.append("id absent from grid")
        elif prompt_want != record.get("prompt_hash"):
            reasons.append("prompt_hash mismatch")

        if config_want is None:
            if prompt_want is not None:
                reasons.append("model absent from config")
        elif config_want != record.get("config_hash"):
            # Rows written before config_hash existed have no key at all,
            # which is a mismatch: their sampling condition is unverifiable.
            reasons.append("config_hash mismatch")

        if reasons:
            stale.append((record, prompt_want, config_want, reasons))
        else:
            fresh.append(record)

    if not stale:
        return fresh

    quarantined_at = _now()
    for record, prompt_want, config_want, reasons in stale:
        append_jsonl(
            STALE_PATH,
            {
                **record,
                "quarantined_at": quarantined_at,
                "quarantine_reason": " + ".join(reasons),
                "grid_prompt_hash": prompt_want,
                "expected_config_hash": config_want,
            },
        )

    # Rewrite via a temp file so an interrupted rewrite cannot truncate the
    # surviving responses; the quarantine copy is already durable by now.
    tmp = RESPONSES_PATH.with_name(RESPONSES_PATH.name + ".tmp")
    with tmp.open("w") as handle:
        for record in fresh:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(RESPONSES_PATH)

    reasons = collections.Counter(" + ".join(r) for *_, r in stale)
    examples = ", ".join(record["id"] for record, *_ in stale[:4])
    more = f" (+{len(stale) - 4} more)" if len(stale) > 4 else ""
    print(
        f"[collect] WARNING: {len(stale)} stale response(s) quarantined -> "
        f"{STALE_PATH.relative_to(ROOT)}",
        file=sys.stderr,
    )
    for reason, count in reasons.most_common():
        print(f"[collect]   {reason}: {count}", file=sys.stderr)
    print(f"[collect]   e.g. {examples}{more}", file=sys.stderr)
    print(
        "[collect]   excluded from the done set; they will be recollected "
        "if the current filters select them",
        file=sys.stderr,
    )
    return fresh


def collect_one(
    item: dict[str, Any],
    model: dict[str, Any],
    decoding: dict[str, Any],
    cfg_hash: str,
) -> bool:
    """Run one grid item and append its result. Returns True on success."""
    provider, name = model["provider"], model["name"]
    with _provider_slots[provider]:
        try:
            result = models.generate(
                provider=provider,
                model=name,
                prompt=item["prompt"],
                temperature=decoding["temperature"],
                top_p=decoding["top_p"],
                max_tokens=decoding["max_tokens"],
            )
        except Exception as exc:  # noqa: BLE001 - a dead call must not stop the run
            append_jsonl(
                FAILURES_PATH,
                {
                    **item,
                    "provider": provider,
                    "model_name": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "ts": _now(),
                },
            )
            return False

    accepted = models.effective_params(provider, name)
    append_jsonl(
        RESPONSES_PATH,
        {
            **item,
            "response": result["text"],
            "model_version": result["model_version"],
            "finish_reason": result["finish_reason"],
            # Pins the row to the sampling condition it was collected under,
            # the way prompt_hash pins it to the prompt text.
            "config_hash": cfg_hash,
            # Requested vs accepted: these differ wherever a provider refused
            # a parameter, so analysis can condition on the real setting.
            "temperature": decoding["temperature"],
            "latency_ms": result["latency_ms"],
            "ts": _now(),
            "effective_temperature": accepted["temperature"],
            "effective_top_p": accepted["top_p"],
            "effective_max_tokens": accepted["max_tokens"],
            "effective_thinking_level": accepted["thinking_level"],
            "effective_effort": accepted["effort"],
            "effective_reasoning_enabled": accepted["reasoning_enabled"],
            "params_dropped": accepted["dropped"],
        },
    )
    return True


def selected(item: dict[str, Any], args: argparse.Namespace) -> bool:
    """True if a grid item passes every active filter.

    Filters conjoin: --model, --issue and --max-run each narrow the set, and
    --limit truncates whatever survives.
    """
    if args.model_ids is not None and item["model_id"] not in args.model_ids:
        return False
    if args.issue is not None and item["issue"] != args.issue:
        return False
    if args.max_run is not None and item["run"] > args.max_run:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="stop after N calls (smoke test)")
    parser.add_argument(
        "--model", help="restrict to one model id or a comma-separated list, e.g. M1,M2"
    )
    parser.add_argument("--issue", help="restrict to one issue, e.g. guns")
    parser.add_argument(
        "--max-run", type=int, metavar="N", help="only include runs numbered <= N"
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    decoding = config["decoding"]
    design = config["design"]
    by_id = {model["id"]: model for model in config["models"]}

    # None means "no model filter"; otherwise a de-duplicated list in the order
    # given, so --model M1 and --model M1,M2 take the same code path.
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
    if args.issue and args.issue not in design["issues"]:
        parser.error(
            f"unknown issue {args.issue!r}; expected one of "
            f"{', '.join(design['issues'])}"
        )
    if args.max_run is not None and args.max_run < 1:
        parser.error("--max-run must be at least 1 (runs are numbered from 1)")

    grid = read_jsonl(GRID_PATH)
    if not grid:
        parser.error(f"{GRID_PATH.relative_to(ROOT)} is empty; run src/prompt_grid.py")

    expected_config = {
        model["id"]: config_hash(model["provider"], model["name"], decoding)
        for model in config["models"]
    }
    responses = quarantine_stale(grid, read_jsonl(RESPONSES_PATH), expected_config)
    done = {record["id"] for record in responses}
    pending = [
        item for item in grid if item["id"] not in done and selected(item, args)
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    for model in config["models"]:
        _provider_slots.setdefault(
            model["provider"], threading.Semaphore(MAX_PER_PROVIDER)
        )

    active = [
        text
        for text in (
            ",".join(args.model_ids) if args.model_ids else None,
            args.issue,
            f"run<={args.max_run}" if args.max_run is not None else None,
            f"limit {args.limit}" if args.limit is not None else None,
        )
        if text
    ]
    scope = f" [{', '.join(active)}]" if active else ""
    print(
        f"{len(grid)} grid items, {len(done)} already collected; "
        f"{len(pending)} to run{scope} "
        f"({MAX_PER_PROVIDER} concurrent per provider)"
    )
    if not pending:
        return 0

    RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    # One worker slot per provider-slot, so the semaphores are the real limit.
    workers = MAX_PER_PROVIDER * len({m["provider"] for m in config["models"]})
    failures = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                collect_one,
                item,
                by_id[item["model_id"]],
                decoding,
                expected_config[item["model_id"]],
            ): item
            for item in pending
        }
        progress = tqdm(as_completed(futures), total=len(futures), unit="call")
        try:
            for future in progress:
                if not future.result():
                    failures += 1
                    progress.set_postfix(failed=failures)
        except KeyboardInterrupt:
            print("\ninterrupted; completed calls are saved, rerun to resume")
            pool.shutdown(wait=False, cancel_futures=True)
            return 130

    collected = len(pending) - failures
    print(f"collected {collected}/{len(pending)}")
    if failures:
        print(
            f"{failures} failed -> {FAILURES_PATH.relative_to(ROOT)} "
            f"(not marked done; rerun to retry)"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
