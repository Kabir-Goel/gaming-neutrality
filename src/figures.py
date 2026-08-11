"""F1-F5 (build guide §9). Reads data/scores/scores.json only.

F1 ADG leaderboard, F2 audit-vs-deploy slopegraph, F3 per-model persona x
frame stance heatmaps, F4 baseline lean vs. steering scatter, F5 probe
detection-gap vs. ADG scatter. F6 (judge-vs-human calibration) is written
by validate.py, not here, since it needs the hand-coding data this script
doesn't touch.

Plain matplotlib (no seaborn styling), Okabe-Ito colorblind-safe palette,
every bar/point that has one gets a CI. Saved to figures/ at 200 dpi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent.parent
SCORES_PATH = ROOT / "data" / "scores" / "scores.json"
FIGDIR = ROOT / "figures"
DPI = 200

# Okabe-Ito: colorblind-safe, distinct at print size.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]

# config.yaml's model.name is the exact API string (e.g.
# "meta-llama/Llama-3.3-70B-Instruct-Turbo") — accurate but too long for
# axis labels. These are the same short forms used in the earlier seaborn
# figures, kept consistent across the whole figure set.
SHORT_NAME = {
    "M1": "claude-sonnet-5",
    "M2": "gpt-5.6-terra",
    "M3": "gemini-3.6-flash",
    "M4": "Llama-3.3-70B",
    "M5": "GLM-5.2",
}


def short_name(model_id: str) -> str:
    return SHORT_NAME.get(model_id, model_id)


def load() -> dict:
    if not SCORES_PATH.exists():
        raise SystemExit(f"{SCORES_PATH.relative_to(ROOT)} not found; run src/score.py first")
    return json.loads(SCORES_PATH.read_text())


def model_color(models: list[dict]) -> dict[str, str]:
    return {m["model_id"]: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def f1_leaderboard(data: dict, colors: dict[str, str]) -> None:
    models = sorted(data["models"], key=lambda m: m["ADG"])
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(models) + 1.4), dpi=DPI)
    y = np.arange(len(models))
    adg = [m["ADG"] for m in models]
    lo = [m["ADG"] - m["ADG_ci"][0] for m in models]
    hi = [m["ADG_ci"][1] - m["ADG"] for m in models]
    bar_colors = [colors[m["model_id"]] for m in models]

    ax.barh(y, adg, xerr=[lo, hi], color=bar_colors, height=0.6, capsize=4, ecolor="#333333")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([short_name(m["model_id"]) for m in models], fontsize=9.5)
    ax.set_xlabel("ADG  (Steering Score, casual frame minus audit frame)")
    ax.set_title(
        "Audit-deployment gap by model\npositive = more persona-driven steering in casual use than under audit",
        fontsize=11,
    )
    for yi, m in enumerate(models):
        label_x = m["ADG_ci"][1] + 0.7 if m["ADG"] >= 0 else m["ADG_ci"][0] - 0.7
        ha = "left" if m["ADG"] >= 0 else "right"
        ax.text(label_x, yi, f"{m['ADG']:+.1f}", va="center", ha=ha, fontsize=8.5, color="#333333")
    ax.set_xlim(min(m["ADG_ci"][0] for m in models) - 3, max(m["ADG_ci"][1] for m in models) + 3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "F1_adg_leaderboard.png", bbox_inches="tight")
    plt.close(fig)


def f2_slopegraph(data: dict, colors: dict[str, str]) -> None:
    models = data["models"]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=DPI)
    x = [0, 1]
    for m in models:
        y = [m["S"]["A"], m["S"]["D"]]
        c = colors[m["model_id"]]
        name = short_name(m["model_id"])
        ax.plot(x, y, color=c, marker="o", linewidth=2, markersize=6)
        ax.text(-0.06, y[0], name, ha="right", va="center", fontsize=9, color=c)
        ax.text(1.06, y[1], f"{name}  ({y[1]:.1f})", ha="left", va="center", fontsize=9, color=c)
    ax.set_xlim(-0.9, 2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["Audit frame", "Casual frame"], fontsize=10)
    ax.set_ylabel("Steering Score S")
    ax.set_title("Steering score, audit vs. casual framing\n(steeper line = larger audit-deployment gap)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "F2_slopegraph.png", bbox_inches="tight")
    plt.close(fig)


def f3_heatmaps(data: dict) -> None:
    models = data["models"]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.4), dpi=DPI, sharey=True)
    if n == 1:
        axes = [axes]

    personas = ["C", "N", "L"]
    frames = ["A", "D"]

    # A cell (persona in {L,C}, frame) is starred if any per-issue L-vs-C
    # Mann-Whitney test for that (model, frame) was BH-significant. This is
    # a coarser granularity than the per-issue test itself (4 tests
    # collapsed to one star), documented here rather than implied.
    sig_by_model_frame: dict[tuple[str, str], bool] = {}
    for t in data.get("per_cell_tests", []):
        key = (t["model_id"], t["frame"])
        sig_by_model_frame[key] = sig_by_model_frame.get(key, False) or t.get("significant_fdr_q10", False)

    vmax = max(
        abs(v) for m in models for persona in personas for v in m["persona_frame_stance"].get(persona, {}).values()
    )
    vmax = max(vmax, 0.05)

    im = None
    for ax, m in zip(axes, models):
        grid = np.array(
            [[m["persona_frame_stance"].get(p, {}).get(f, np.nan) for f in frames] for p in personas]
        )
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i, p in enumerate(personas):
            for j, f in enumerate(frames):
                val = grid[i, j]
                star = "*" if p in ("L", "C") and sig_by_model_frame.get((m["model_id"], f)) else ""
                ax.text(j, i, f"{val:+.2f}{star}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(val) > vmax * 0.6 else "black")
        ax.set_xticks(range(len(frames)))
        ax.set_xticklabels(["Audit", "Casual"], fontsize=8.5)
        ax.set_yticks(range(len(personas)))
        ax.set_yticklabels(personas, fontsize=9)
        ax.set_title(m["model_id"], fontsize=10)

    fig.suptitle(
        "Mean stance by persona x frame, per model  (* = any issue BH-significant, L vs C, q=0.10)",
        fontsize=10.5, y=1.04,
    )
    fig.colorbar(im, ax=axes, shrink=0.7, label="mean stance", pad=0.02)
    fig.savefig(FIGDIR / "F3_heatmap_per_model.png", bbox_inches="tight")
    plt.close(fig)


def f4_lean_vs_steering(data: dict, colors: dict[str, str]) -> None:
    models = data["models"]
    xs, ys, labels = [], [], []
    for m in models:
        lean_vals = list(m["baseline_lean"]["D"].values())
        x = float(np.mean([abs(v) for v in lean_vals]))
        y = m["S"]["D"]
        xs.append(x)
        ys.append(y)
        labels.append(m["model_id"])

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=DPI)
    for x, y, lbl in zip(xs, ys, labels):
        name = short_name(lbl)
        ax.scatter(x, y, color=colors[lbl], s=90, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9, color=colors[lbl])

    ax.axvline(float(np.median(xs)), color="grey", linewidth=0.8, linestyle="--")
    ax.axhline(float(np.median(ys)), color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("|baseline lean| (persona = N, casual frame, avg. over issues)")
    ax.set_ylabel("steering (Steering Score, casual frame)")
    ax.set_title(
        "A model's default-voice lean and its persona-steering\nare different things (dashed lines = median)",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "F4_lean_vs_steering.png", bbox_inches="tight")
    plt.close(fig)


def f5_probe_vs_adg(data: dict, colors: dict[str, str]) -> None:
    models = data["models"]
    xs, ys, labels = [], [], []
    for m in models:
        rate = m.get("probe_eval_rate", {})
        if "A" not in rate or "D" not in rate or rate["A"] is None or rate["D"] is None:
            continue
        xs.append(rate["A"] - rate["D"])
        ys.append(m["ADG"])
        labels.append(m["model_id"])

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=DPI)
    for x, y, lbl in zip(xs, ys, labels):
        name = short_name(lbl)
        ax.scatter(x, y, color=colors[lbl], s=90, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9, color=colors[lbl])

    if len(xs) >= 3:
        reg = linregress(xs, ys)
        xline = np.linspace(min(xs), max(xs), 50)
        ax.plot(xline, reg.intercept + reg.slope * xline, color="#333333", linewidth=1, linestyle="--")
        ax.text(
            0.03, 0.95, f"r = {reg.rvalue:.2f}", transform=ax.transAxes, fontsize=10, va="top",
        )

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("probe eval-rate gap  (P(self-report \"evaluation\") | audit  -  | casual)")
    ax.set_ylabel("ADG")
    ax.set_title("Does detecting the audit predict steering less under it?", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "F5_probe_vs_adg.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    data = load()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    colors = model_color(data["models"])

    f1_leaderboard(data, colors)
    print("wrote figures/F1_adg_leaderboard.png")
    f2_slopegraph(data, colors)
    print("wrote figures/F2_slopegraph.png")
    f3_heatmaps(data)
    print("wrote figures/F3_heatmap_per_model.png")
    f4_lean_vs_steering(data, colors)
    print("wrote figures/F4_lean_vs_steering.png")
    f5_probe_vs_adg(data, colors)
    print("wrote figures/F5_probe_vs_adg.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
