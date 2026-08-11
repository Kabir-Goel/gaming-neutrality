"""The judge validation gate (build guide §7.5) — do not skip this.

Reads the completed data/coded/to_hand_code.csv (produced blind by
scripts/make_hand_code_sample.py and filled in by hand, one sitting, no
peeking at judge scores), rejoins it to the real ids via the gitignored
data/coded/hand_code_key.jsonl, and compares human scores to the judge's
scores on the same rows:

- ICC(2,1) absolute agreement for stance and framing (pingouin)
- Quadratic-weighted Cohen's kappa for refusal (sklearn)
- Bootstrap 95% CI on each, resampling rows with replacement
- A judge-vs-human calibration scatter for stance -> figures/F6_calibration.png

Also writes the de-blinded data/coded/human_codes.jsonl (id + human scores,
no code column) — this is the artifact named in the build guide, safe to
commit since coding is finished by the time it exists.

Prints PASS/FAIL against config.validation.thresholds and exits 1 on any
failure, so it can gate a CI step or just your own attention. A failure
here is not a bug to hide — the guide is explicit that an honest reported
disagreement is worth more than a silently accepted one.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import yaml
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
TO_HAND_CODE = ROOT / "data" / "coded" / "to_hand_code.csv"
KEY_PATH = ROOT / "data" / "coded" / "hand_code_key.jsonl"
CODED_PATH = ROOT / "data" / "coded" / "coded.jsonl"
HUMAN_CODES_PATH = ROOT / "data" / "coded" / "human_codes.jsonl"
OUT_PATH = ROOT / "data" / "scores" / "validation.json"
FIGURE_PATH = ROOT / "figures" / "F6_calibration.png"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_human_scores() -> pd.DataFrame:
    if not TO_HAND_CODE.exists():
        raise SystemExit(
            f"{TO_HAND_CODE.relative_to(ROOT)} not found; "
            "run scripts/make_hand_code_sample.py first"
        )
    if not KEY_PATH.exists():
        raise SystemExit(
            f"{KEY_PATH.relative_to(ROOT)} not found (it's gitignored — "
            "make sure you're running this from the machine that generated "
            "the sample, or regenerate it, which only works before any "
            "hand-coding has started)"
        )

    with TO_HAND_CODE.open(newline="") as f:
        rows = list(csv.DictReader(f))

    unfilled = [r["code"] for r in rows if not all(r[c].strip() for c in ("stance", "framing", "refusal"))]
    if unfilled:
        raise SystemExit(
            f"{len(unfilled)}/{len(rows)} rows in to_hand_code.csv still have "
            f"empty stance/framing/refusal (e.g. code {unfilled[0]}) — "
            "finish hand-coding before running validate.py"
        )

    # JSON round-trips the code as an int; csv.DictReader hands back every
    # field as a str regardless of what was written, so the lookup has to
    # normalize one side or every code silently fails to match.
    key = {r["code"]: r["id"] for r in read_jsonl(KEY_PATH)}
    missing_key = [r["code"] for r in rows if int(r["code"]) not in key]
    if missing_key:
        raise SystemExit(f"{len(missing_key)} codes in the CSV have no entry in hand_code_key.jsonl")

    return pd.DataFrame(
        [
            {
                "id": key[int(r["code"])],
                "human_stance": float(r["stance"]),
                "human_framing": float(r["framing"]),
                "human_refusal": int(float(r["refusal"])),
            }
            for r in rows
        ]
    )


def join_judge_scores(human: pd.DataFrame) -> pd.DataFrame:
    judge = {r["id"]: r for r in read_jsonl(CODED_PATH) if r.get("parse_ok")}
    missing = [i for i in human["id"] if i not in judge]
    if missing:
        raise SystemExit(
            f"{len(missing)} hand-coded ids have no parsed judge score "
            f"(e.g. {missing[0]}) — re-run src/judge.py"
        )
    human = human.copy()
    human["judge_stance"] = human["id"].map(lambda i: judge[i]["stance"])
    human["judge_framing"] = human["id"].map(lambda i: judge[i]["framing"])
    human["judge_refusal"] = human["id"].map(lambda i: judge[i]["refusal"])
    return human


def icc21(df: pd.DataFrame, human_col: str, judge_col: str) -> float:
    """ICC(2,1): two-way random effects, single rater, absolute agreement."""
    long = pd.concat(
        [
            pd.DataFrame({"target": df["id"], "rater": "human", "rating": df[human_col].to_numpy()}),
            pd.DataFrame({"target": df["id"], "rater": "judge", "rating": df[judge_col].to_numpy()}),
        ],
        ignore_index=True,
    )
    result = pg.intraclass_corr(data=long, targets="target", raters="rater", ratings="rating")
    # pingouin >=0.5 renamed the classic Shrout & Fleiss ICC2 (two-way
    # random, absolute agreement, single rater) to "ICC(A,1)" — "ICC2" is
    # not a value this version ever produces.
    row = result[result["Type"] == "ICC(A,1)"]
    return float(row["ICC"].iloc[0])


def bootstrap_ci(values: np.ndarray, stat_fn, iters: int, seed: int) -> tuple[float, list[float]]:
    """Percentile bootstrap. A resample can be degenerate (e.g. every human
    refusal rating the same value, which makes weighted kappa's expected-
    agreement term zero) — that raises on some inputs and silently returns
    nan on others depending on the metric, so both cases are filtered out
    rather than being allowed to poison the percentile with a single nan.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        try:
            value = stat_fn(idx)
        except Exception:  # noqa: BLE001 - a degenerate resample just gets skipped
            continue
        if np.isfinite(value):
            draws.append(value)
    dropped = iters - len(draws)
    if dropped:
        print(f"  ({dropped}/{iters} bootstrap draws were degenerate and skipped)")
    if not draws:
        return float("nan"), [float("nan"), float("nan")]
    return float(np.mean(draws)), [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> int:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    thresholds = config["validation"]["thresholds"]
    iters = config["scoring"]["bootstrap_iters"]
    seed = config["seed"]

    human = load_human_scores()
    df = join_judge_scores(human)
    n = len(df)
    print(f"{n} hand-coded rows joined to judge scores")

    HUMAN_CODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HUMAN_CODES_PATH.open("w") as f:
        for _, row in df.iterrows():
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "stance": row["human_stance"],
                        "framing": row["human_framing"],
                        "refusal": int(row["human_refusal"]),
                    }
                )
                + "\n"
            )
    print(f"wrote {HUMAN_CODES_PATH.relative_to(ROOT)} ({n} rows)")

    stance_icc = icc21(df, "human_stance", "judge_stance")
    framing_icc = icc21(df, "human_framing", "judge_framing")
    refusal_kappa = cohen_kappa_score(
        df["human_refusal"].to_numpy(), df["judge_refusal"].to_numpy(), weights="quadratic"
    )

    stance_arr = df[["human_stance", "judge_stance"]].reset_index(drop=True)
    framing_arr = df[["human_framing", "judge_framing"]].reset_index(drop=True)
    refusal_arr = df[["human_refusal", "judge_refusal"]].reset_index(drop=True)

    def stance_stat(idx):
        sub = stance_arr.iloc[idx].reset_index(drop=True)
        sub_ids = pd.Series(range(len(sub)))
        return icc21(
            pd.DataFrame({"id": sub_ids, "human_stance": sub["human_stance"], "judge_stance": sub["judge_stance"]}),
            "human_stance", "judge_stance",
        )

    def framing_stat(idx):
        sub = framing_arr.iloc[idx].reset_index(drop=True)
        sub_ids = pd.Series(range(len(sub)))
        return icc21(
            pd.DataFrame({"id": sub_ids, "human_framing": sub["human_framing"], "judge_framing": sub["judge_framing"]}),
            "human_framing", "judge_framing",
        )

    def refusal_stat(idx):
        sub = refusal_arr.iloc[idx]
        return cohen_kappa_score(sub["human_refusal"], sub["judge_refusal"], weights="quadratic")

    print(f"bootstrapping CIs ({iters} iters)...")
    _, stance_ci = bootstrap_ci(np.arange(n), stance_stat, iters, seed)
    _, framing_ci = bootstrap_ci(np.arange(n), framing_stat, iters, seed + 1)
    _, refusal_ci = bootstrap_ci(np.arange(n), refusal_stat, iters, seed + 2)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.scatter(df["judge_stance"], df["human_stance"], alpha=0.5, s=20, color="#4C72B0")
    ax.plot([-1, 1], [-1, 1], color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("judge stance")
    ax.set_ylabel("human stance")
    ax.set_title(f"Judge vs. human, stance (n={n}, ICC2={stance_icc:.2f})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)
    print(f"wrote {FIGURE_PATH.relative_to(ROOT)}")

    checks = [
        ("stance_icc", stance_icc, stance_ci, thresholds["stance_icc"]),
        ("framing_icc", framing_icc, framing_ci, thresholds["framing_icc"]),
        ("refusal_kappa", refusal_kappa, refusal_ci, thresholds["refusal_kappa"]),
    ]

    print(f"\n{'metric':<16} {'value':>8} {'95% CI':>18} {'threshold':>10} {'result':>8}")
    all_pass = True
    for name, value, ci, threshold in checks:
        passed = value >= threshold
        all_pass &= passed
        print(
            f"{name:<16} {value:>8.3f} [{ci[0]:>6.3f}, {ci[1]:>6.3f}] "
            f"{threshold:>10.2f} {'PASS' if passed else 'FAIL':>8}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(
            {
                "n": n,
                "stance_icc": stance_icc,
                "stance_icc_ci": stance_ci,
                "framing_icc": framing_icc,
                "framing_icc_ci": framing_ci,
                "refusal_kappa": refusal_kappa,
                "refusal_kappa_ci": refusal_ci,
                "thresholds": thresholds,
                "pass": all_pass,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}")
    print("GATE: PASS" if all_pass else "GATE: FAIL")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
