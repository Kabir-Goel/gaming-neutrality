"""Draw the blinded human-coding sample for judge validation (build guide §7.4).

Samples config.validation.sample_frac of coded.jsonl, seeded from
config.seed, stratified by model: each model is sampled at the same
sample_frac independently, rather than sample_frac drawn once from the
pooled 1920 rows. An unstratified draw can land anywhere from ~37 to ~52
rows for a given model by chance; stratifying pins every model to ~46 (12%
of its 384 rows) so a model that happens to carry most of the headline
effect — gpt-5.6-terra does here — isn't left thinly validated by luck of
the draw.

Each sampled row gets an opaque numeric code with no relationship to its
real id — the real id (`{issue}__{persona}__{frame}__{model}__r{run}`)
spells out all four design factors, so using it as the row label would let
a human coder infer persona/frame/model just from the label, defeating the
point of blind coding.

Writes two files:
  data/coded/to_hand_code.csv   — code, response_text, stance, framing,
                                   refusal (last three empty; this is what
                                   gets hand-coded)
  data/coded/hand_code_key.jsonl — code -> id, gitignored. Never open this
                                    while hand-coding; it exists only so
                                    validate.py can rejoin scores to the
                                    real rows afterward.

Idempotent in the sense that it always regenerates from the same seed, so
re-running before any hand-coding has started reproduces the same sample.
Refuses to overwrite an existing to_hand_code.csv that already has any
non-empty stance/framing/refusal cells, so a re-run can't clobber real
in-progress coding.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODED_PATH = ROOT / "data" / "coded" / "coded.jsonl"
RESPONSES_PATH = ROOT / "data" / "raw" / "responses.jsonl"
OUT_CSV = ROOT / "data" / "coded" / "to_hand_code.csv"
OUT_KEY = ROOT / "data" / "coded" / "hand_code_key.jsonl"

LEAKY_COLUMNS = {"issue", "persona", "frame", "model_id", "model", "run", "id"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def existing_sample_has_progress(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if any(row.get(k, "").strip() for k in ("stance", "framing", "refusal")):
                return True
    return False


def main() -> int:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    seed = config["seed"]
    frac = config["validation"]["sample_frac"]

    if existing_sample_has_progress(OUT_CSV):
        print(
            f"{OUT_CSV.relative_to(ROOT)} already has hand-coded values in it — "
            "refusing to regenerate and overwrite in-progress work. "
            "Delete it yourself first if you really want a fresh sample.",
            file=sys.stderr,
        )
        return 1

    coded = read_jsonl(CODED_PATH)
    if not coded:
        print(f"{CODED_PATH.relative_to(ROOT)} is empty; run src/judge.py first", file=sys.stderr)
        return 1
    coded = [row for row in coded if row.get("parse_ok")]

    responses = {row["id"]: row for row in read_jsonl(RESPONSES_PATH)}
    missing = [row["id"] for row in coded if row["id"] not in responses]
    if missing:
        print(
            f"{len(missing)} coded ids have no matching response "
            f"(e.g. {missing[0]}); is responses.jsonl complete?",
            file=sys.stderr,
        )
        return 1

    rng = np.random.default_rng(seed)

    # Stratify by model_id, in config order, so the draw is reproducible
    # regardless of coded.jsonl's on-disk row order.
    model_ids = [m["id"] for m in config["models"]]
    by_model: dict[str, list[dict]] = {mid: [] for mid in model_ids}
    for row in coded:
        by_model[responses[row["id"]]["model_id"]].append(row)

    sample = []
    for mid in model_ids:
        group = by_model[mid]
        n_model = round(len(group) * frac)
        idx = rng.choice(len(group), size=n_model, replace=False)
        sample.extend(group[i] for i in idx)
    n = len(sample)

    # Opaque codes: a random permutation of a 6-digit range, independent of
    # sample order and of id. No arithmetic relationship to row position,
    # issue/persona/frame/model, or draw order.
    codes = rng.choice(900000, size=n, replace=False) + 100000
    rows = [
        {
            "code": int(code),
            "response_text": responses[row["id"]]["response"],
            "id": row["id"],
        }
        for code, row in zip(codes, sample)
    ]
    rng.shuffle(rows)  # break any residual order correlation with the draw

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "response_text", "stance", "framing", "refusal"])
        for row in rows:
            writer.writerow([row["code"], row["response_text"], "", "", ""])

    with OUT_KEY.open("w") as f:
        for row in rows:
            f.write(json.dumps({"code": row["code"], "id": row["id"]}) + "\n")

    leaked = LEAKY_COLUMNS & set(next(csv.reader(OUT_CSV.open())))
    print(f"sample size: {n} / {len(coded)} coded rows ({frac:.0%}), stratified by model:")
    for mid in model_ids:
        print(f"  {mid}: {round(len(by_model[mid]) * frac)} / {len(by_model[mid])}")
    print(f"wrote {OUT_CSV.relative_to(ROOT)} (columns: code, response_text, stance, framing, refusal)")
    print(f"wrote {OUT_KEY.relative_to(ROOT)} (gitignored — do not open while hand-coding)")
    print(f"leaked design-factor columns in the CSV: {leaked or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
