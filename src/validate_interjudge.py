"""Inter-judge reliability: Opus-5 (the real judge) vs. Haiku (a second,
independent judge), in place of the build guide's §7.5 human-coding gate.

This is a deliberate substitution, not the guide's design. Recorded in
notes/decisions.md: hand-coding ~230 responses was cut for personal-bias
risk and time cost, and inter-judge agreement is used instead, with the
absence of human validation reported as a limitation. Keep that framing
wherever these numbers get quoted — ICC/kappa between two LLM judges shows
whether Opus and Haiku *agree with each other*, not whether either is
right. It is evidence the rubric is precise enough to be shared across
models, not evidence the scores match human judgment.

Compares data/coded/coded.jsonl (Opus) to data/coded/coded_haiku.jsonl
(Haiku) on every id both parsed successfully, computes the same three
metrics as validate.py (ICC(2,1) for stance/framing, quadratic-weighted
kappa for refusal, each with a bootstrap 95% CI), and writes
data/scores/interjudge_agreement.json — a different file from
validation.json so the two can never be confused for each other.
"""

from __future__ import annotations

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
OPUS_PATH = ROOT / "data" / "coded" / "coded.jsonl"
HAIKU_PATH = ROOT / "data" / "coded" / "coded_haiku.jsonl"
RESPONSES_PATH = ROOT / "data" / "raw" / "responses.jsonl"
OUT_PATH = ROOT / "data" / "scores" / "interjudge_agreement.json"
FIGURE_PATH = ROOT / "figures" / "F6_interjudge_calibration.png"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} not found")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def icc21(df: pd.DataFrame, a_col: str, b_col: str) -> float:
    long = pd.concat(
        [
            pd.DataFrame({"target": df["id"], "rater": "opus", "rating": df[a_col].to_numpy()}),
            pd.DataFrame({"target": df["id"], "rater": "haiku", "rating": df[b_col].to_numpy()}),
        ],
        ignore_index=True,
    )
    result = pg.intraclass_corr(data=long, targets="target", raters="rater", ratings="rating")
    # pingouin >=0.5 renamed classic ICC2 (two-way random, absolute
    # agreement, single rater) to "ICC(A,1)".
    row = result[result["Type"] == "ICC(A,1)"]
    return float(row["ICC"].iloc[0])


def bootstrap_ci(n: int, stat_fn, iters: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        try:
            value = stat_fn(idx)
        except Exception:  # noqa: BLE001 - a degenerate resample just gets skipped
            continue
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return [float("nan"), float("nan")]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> int:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    thresholds = config["validation"]["thresholds"]
    iters = config["scoring"]["bootstrap_iters"]
    seed = config["seed"]

    opus = {r["id"]: r for r in read_jsonl(OPUS_PATH) if r.get("parse_ok")}
    haiku = {r["id"]: r for r in read_jsonl(HAIKU_PATH) if r.get("parse_ok")}
    responses = {r["id"]: r for r in read_jsonl(RESPONSES_PATH)}

    shared_ids = sorted(set(opus) & set(haiku))
    if not shared_ids:
        raise SystemExit("no ids are judged by both coded.jsonl and coded_haiku.jsonl")

    coverage_note = None
    if len(shared_ids) < len(responses):
        by_model: dict[str, int] = {}
        for i in shared_ids:
            mid = responses.get(i, {}).get("model_id", "?")
            by_model[mid] = by_model.get(mid, 0) + 1
        coverage_note = (
            f"{len(shared_ids)}/{len(responses)} responses have a Haiku score "
            f"(partial run) — per-model: {by_model}"
        )
        print(f"NOTE: {coverage_note}")

    df = pd.DataFrame(
        [
            {
                "id": i,
                "opus_stance": opus[i]["stance"],
                "haiku_stance": haiku[i]["stance"],
                "opus_framing": opus[i]["framing"],
                "haiku_framing": haiku[i]["framing"],
                "opus_refusal": opus[i]["refusal"],
                "haiku_refusal": haiku[i]["refusal"],
            }
            for i in shared_ids
        ]
    )
    n = len(df)
    print(f"{n} responses judged by both Opus and Haiku")

    stance_icc = icc21(df, "opus_stance", "haiku_stance")
    framing_icc = icc21(df, "opus_framing", "haiku_framing")
    refusal_kappa = cohen_kappa_score(
        df["opus_refusal"].to_numpy(), df["haiku_refusal"].to_numpy(), weights="quadratic"
    )

    def stance_stat(idx):
        sub = df.iloc[idx].reset_index(drop=True)
        sub = sub.assign(id=range(len(sub)))
        return icc21(sub, "opus_stance", "haiku_stance")

    def framing_stat(idx):
        sub = df.iloc[idx].reset_index(drop=True)
        sub = sub.assign(id=range(len(sub)))
        return icc21(sub, "opus_framing", "haiku_framing")

    def refusal_stat(idx):
        sub = df.iloc[idx]
        return cohen_kappa_score(sub["opus_refusal"], sub["haiku_refusal"], weights="quadratic")

    print(f"bootstrapping CIs ({iters} iters)...")
    stance_ci = bootstrap_ci(n, stance_stat, iters, seed)
    framing_ci = bootstrap_ci(n, framing_stat, iters, seed + 1)
    refusal_ci = bootstrap_ci(n, refusal_stat, iters, seed + 2)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.scatter(df["opus_stance"], df["haiku_stance"], alpha=0.4, s=18, color="#4C72B0")
    ax.plot([-1, 1], [-1, 1], color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Opus-5 stance (the real judge)")
    ax.set_ylabel("Haiku stance (second judge)")
    ax.set_title(f"Inter-judge agreement, stance (n={n}, ICC2={stance_icc:.2f})\nNOT validated against a human coder", fontsize=10)
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
    print(
        "\n(thresholds below are the config.yaml human-validation thresholds, reused "
        "here only as a rough reference point — this run never touched a human coder)"
    )
    print(f"\n{'metric':<16} {'value':>8} {'95% CI':>18} {'threshold':>10} {'vs. threshold':>14}")
    all_pass = True
    for name, value, ci, threshold in checks:
        passed = value >= threshold
        all_pass &= passed
        print(
            f"{name:<16} {value:>8.3f} [{ci[0]:>6.3f}, {ci[1]:>6.3f}] "
            f"{threshold:>10.2f} {'meets' if passed else 'below':>14}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(
            {
                "method": "inter_judge_opus_vs_haiku",
                "not_human_validated": True,
                "n": n,
                "coverage_note": coverage_note,
                "stance_icc": stance_icc,
                "stance_icc_ci": stance_ci,
                "framing_icc": framing_icc,
                "framing_icc_ci": framing_ci,
                "refusal_kappa": refusal_kappa,
                "refusal_kappa_ci": refusal_ci,
                "thresholds_reused_from_config": thresholds,
                "meets_thresholds": all_pass,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}")
    print(
        "This is inter-judge agreement (Opus vs. Haiku), not judge-vs-human "
        "validation. See notes/decisions.md."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
