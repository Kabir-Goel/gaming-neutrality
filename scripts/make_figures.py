"""Generate publication figures for the gaming-neutrality evidence checkpoint.

Reads data/coded/coded.jsonl, recovers the design factors from the deterministic
id ({issue}__{persona}__{frame}__{model}__r{run}), and writes figures/*.png|pdf.

Run: python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "coded" / "coded.jsonl"
FIGDIR = ROOT / "figures"

SEED = 20260805
N_BOOT = 10_000

# --- shared style -----------------------------------------------------------

# One restrained palette, reused for the same category in every figure.
AUDIT = "#3D5A80"  # frame A, formal "federal compliance assessment"
CASUAL = "#E07A3F"  # frame D, casual "texting a friend"
POOLED = "#8A9BA8"  # all models / all frames
STANCE = "#3D5A80"
FRAMING = "#8E5572"

FRAME_COLORS = {"A": AUDIT, "D": CASUAL}
FRAME_LABELS = {"A": "Audit (A)", "D": "Casual (D)"}

MODEL_NAMES = {
    "M1": "claude-sonnet-5",
    "M2": "gpt-5.6-terra",
    "M3": "gemini-3.6-flash",
    "M4": "Llama-3.3-70B",
    "M5": "GLM-5.2",
}
# Two-line versions so grouped-bar axes stay legible without rotated ticks.
MODEL_WRAPPED = {
    "M1": "claude-\nsonnet-5",
    "M2": "gpt-\n5.6-terra",
    "M3": "gemini-\n3.6-flash",
    "M4": "Llama-\n3.3-70B",
    "M5": "GLM-5.2",
}
MODEL_ORDER = ["M1", "M2", "M3", "M4", "M5"]
PERSONA_ORDER = ["C", "N", "L"]
ISSUE_ORDER = ["abortion", "climate", "guns", "immigration"]

INK = "#2B2B2B"
MUTED = "#8C8C8C"
HAIRLINE = "#D6D6D6"

NOTE_KW = dict(fontsize=7, color=MUTED, ha="left", va="bottom")


def set_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "medium",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": HAIRLINE,
            "axes.linewidth": 0.8,
            "text.color": INK,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.size": 0,
            "ytick.major.size": 3,
            "ytick.major.width": 0.8,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def load() -> pd.DataFrame:
    df = pd.read_json(DATA, lines=True)
    parts = df["id"].str.split("__", expand=True)
    parts.columns = ["issue", "persona", "frame", "model", "run"]
    df = pd.concat([df, parts], axis=1)

    # Sanity: the design in config.yaml is 4 x 3 x 2 x 5 x 16 = 1920, fully crossed.
    cell_sizes = df.groupby(["issue", "persona", "frame", "model"]).size()
    assert len(df) == 1920, f"expected 1920 rows, got {len(df)}"
    assert set(cell_sizes) == {16}, f"unbalanced cells: {sorted(set(cell_sizes))}"
    assert df[["stance", "framing", "refusal"]].isna().sum().sum() == 0
    return df


# --- bootstrap helpers ------------------------------------------------------


def boot_mean_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    draws = rng.choice(x, size=(N_BOOT, x.size), replace=True).mean(axis=1)
    return float(x.mean()), *np.percentile(draws, [2.5, 97.5])


def boot_diff_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean(a) - Mean(b), 95% percentile bootstrap, resampled independently."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    da = rng.choice(a, size=(N_BOOT, a.size), replace=True).mean(axis=1)
    db = rng.choice(b, size=(N_BOOT, b.size), replace=True).mean(axis=1)
    d = da - db
    return float(a.mean() - b.mean()), *np.percentile(d, [2.5, 97.5])


def align_suptitle(fig: plt.Figure, ax: plt.Axes) -> None:
    """Left-align a figure-level title with the plotting area.

    Single-axes figures use ax.set_title(loc="left"), which anchors to the axes
    box. fig.suptitle anchors to the figure instead, so multi-panel figures need
    their title nudged to the leftmost axes to match. Call after tight_layout.
    """
    if fig._suptitle is None:
        return
    fig._suptitle.set_x(ax.get_position().x0)
    fig._suptitle.set_horizontalalignment("left")


def save(fig: plt.Figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.png")
    fig.savefig(FIGDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote figures/{name}.png and .pdf")


# --- figure 1 ---------------------------------------------------------------


def fig_persona_effect(df: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for metric, color in (("stance", STANCE), ("framing", FRAMING)):
        for persona in PERSONA_ORDER:
            vals = df.loc[df.persona == persona, metric].to_numpy()
            m, lo, hi = boot_mean_ci(vals, rng)
            rows.append(
                dict(metric=metric, persona=persona, mean=m, lo=lo, hi=hi,
                     n=vals.size, color=color)
            )
    est = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhline(0, color=HAIRLINE, lw=0.8, zorder=1)

    offsets = {"stance": -0.10, "framing": 0.10}
    xpos = {p: i for i, p in enumerate(PERSONA_ORDER)}
    for metric, marker in (("stance", "o"), ("framing", "s")):
        sub = est[est.metric == metric]
        x = np.array([xpos[p] + offsets[metric] for p in sub.persona])
        ax.vlines(x, sub.lo, sub.hi, color=sub.color.iloc[0], lw=1.8, zorder=2)
        ax.scatter(
            x, sub["mean"], s=34, marker=marker, color=sub.color.iloc[0],
            zorder=3, edgecolor="white", linewidth=0.7,
        )
        for xi, mi, lo, hi in zip(x, sub["mean"], sub.lo, sub.hi):
            # Label above the interval for positive means, below for negative,
            # so nothing collides with the zero line.
            anchor, dy, va = (hi, 5, "bottom") if mi >= 0 else (lo, -6, "top")
            ax.annotate(
                f"{mi:+.2f}", (xi, anchor), textcoords="offset points",
                xytext=(0, dy), ha="center", va=va, fontsize=8,
                color=sub.color.iloc[0],
            )

    ax.set_xticks(range(len(PERSONA_ORDER)))
    ax.set_xticklabels(PERSONA_ORDER)
    ax.set_xlim(-0.45, len(PERSONA_ORDER) - 0.55)
    ax.set_ylim(-0.09, 0.25)
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_ylabel("mean score  (+ = progressive)")
    ax.set_title("Stance and framing by persona", loc="left", pad=14)
    ax.legend(
        handles=[
            Line2D([], [], color=STANCE, marker="o", ls="", ms=5, label="stance"),
            Line2D([], [], color=FRAMING, marker="s", ls="", ms=5, label="framing"),
        ],
        loc="lower right", ncol=1,
    )
    fig.text(
        0.005, -0.07,
        "C conservative · N none · L progressive · n=640 each · 95% bootstrap CI",
        **NOTE_KW,
    )
    sns.despine(ax=ax, bottom=True)
    save(fig, "persona_effect")


# --- figure 2 ---------------------------------------------------------------


def fig_adg_by_model(df: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    groups = [(m, df[df.model == m]) for m in MODEL_ORDER] + [("ALL", df)]
    for key, sub in groups:
        for frame in ("A", "D"):
            f = sub[sub.frame == frame]
            d, lo, hi = boot_diff_ci(
                f.loc[f.persona == "L", "stance"].to_numpy(),
                f.loc[f.persona == "C", "stance"].to_numpy(),
                rng,
            )
            rows.append(dict(group=key, frame=frame, spread=d, lo=lo, hi=hi))
    est = pd.DataFrame(rows)

    labels = [MODEL_WRAPPED[m] for m in MODEL_ORDER] + ["pooled"]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0, color=HAIRLINE, lw=0.8, zorder=1)

    for i, frame in enumerate(("A", "D")):
        sub = est[est.frame == frame].set_index("group").loc[MODEL_ORDER + ["ALL"]]
        xi = x + (i - 0.5) * width
        # Pooled group is drawn lighter so it reads as a summary, not a sixth model.
        colors = [to_rgba(FRAME_COLORS[frame], 1.0)] * len(MODEL_ORDER)
        colors += [to_rgba(FRAME_COLORS[frame], 0.45)]
        bars = ax.bar(
            xi, sub.spread, width, color=colors, zorder=2,
            label=FRAME_LABELS[frame], linewidth=0,
        )
        ax.vlines(xi, sub.lo, sub.hi, color=INK, lw=0.9, zorder=3)
        for b, v, hi in zip(bars, sub.spread, sub.hi):
            ax.annotate(
                f"{v:.2f}", (b.get_x() + b.get_width() / 2, max(hi, v)),
                textcoords="offset points", xytext=(0, 3), ha="center",
                fontsize=7.5, color=MUTED,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("stance spread  (L − C)")
    ax.set_ylim(top=0.66)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_title("Persona stance spread by model and frame", loc="left", pad=14)
    ax.legend(loc="upper right", ncol=1)
    fig.text(
        0.005, -0.16,
        "A audit · D casual · n=64 per cell (320 pooled) · 95% bootstrap CI",
        **NOTE_KW,
    )
    sns.despine(ax=ax, bottom=True)
    save(fig, "adg_by_model")


# --- figure 3 ---------------------------------------------------------------


def fig_hedge_rate(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)

    panels = [
        (axes[0], "issue", ISSUE_ORDER, [i.capitalize() for i in ISSUE_ORDER]),
        (axes[1], "model", MODEL_ORDER, [MODEL_WRAPPED[m] for m in MODEL_ORDER]),
    ]
    width = 0.36
    for ax, col, order, labels in panels:
        rate = df.groupby([col, "frame"]).refusal.mean().unstack().loc[order]
        x = np.arange(len(order))
        for i, frame in enumerate(("A", "D")):
            bars = ax.bar(
                x + (i - 0.5) * width, rate[frame], width,
                color=FRAME_COLORS[frame], label=FRAME_LABELS[frame],
                linewidth=0, zorder=2,
            )
            for b, v in zip(bars, rate[frame]):
                # Label every bar, including true zeros, so an empty slot is
                # never mistaken for missing data.
                ax.annotate(
                    f"{v:.0%}" if v >= 0.005 else "0",
                    (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=7.5, color=MUTED if v >= 0.005 else HAIRLINE,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        sns.despine(ax=ax, bottom=True)

    axes[0].set_ylabel("hedge rate")
    axes[0].set_ylim(0, 0.55)
    axes[0].set_yticks([0, 0.25, 0.5])
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    # Keep the legend inside the axes; anchoring it above makes tight_layout
    # reserve a large empty band under the title.
    axes[0].legend(loc="upper right", ncol=1)
    # Right panel shares the y scale; drop its bare spine and tick marks.
    sns.despine(ax=axes[1], left=True, bottom=True)
    axes[1].tick_params(axis="y", length=0)

    fig.suptitle(
        "Hedge rate by issue and model, split by frame",
        ha="left", fontsize=10.5, fontweight="medium", color=INK, y=1.0,
    )
    fig.text(
        0.005, -0.17,
        "A audit · D casual · refusal ≥ 1 · n=240 per issue, 192 per model",
        **NOTE_KW,
    )
    fig.tight_layout()
    align_suptitle(fig, axes[0])
    save(fig, "hedge_rate_by_frame")


# --- figure 4 ---------------------------------------------------------------


def fig_stance_framing_scatter(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.6), sharex=True, sharey=True)
    flat = axes.ravel()

    for ax, m in zip(flat, MODEL_ORDER):
        sub = df[df.model == m]
        ax.axhline(0, color=HAIRLINE, lw=0.7, zorder=0)
        ax.axvline(0, color=HAIRLINE, lw=0.7, zorder=0)
        for frame in ("A", "D"):
            f = sub[sub.frame == frame]
            ax.scatter(
                f.stance, f.framing, s=11, alpha=0.4, linewidth=0,
                color=FRAME_COLORS[frame], label=FRAME_LABELS[frame], zorder=2,
            )
        ax.set_title(MODEL_NAMES[m], fontsize=9.5, pad=6)
        # Bottom-right is empty in every panel; top-left collides with the frame.
        ax.annotate(
            f"r {sub.stance.corr(sub.framing):.2f}   SD {sub.stance.std():.2f}",
            (0.97, 0.04), xycoords="axes fraction", va="bottom", ha="right",
            fontsize=7.5, color=MUTED,
        )
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_box_aspect(1)  # square panels without fighting tight_layout
        sns.despine(ax=ax)

    # Sixth cell carries the legend rather than an empty frame.
    legend_ax = flat[-1]
    legend_ax.axis("off")
    # Nothing sits below the top-right panel, so sharex would otherwise leave it
    # without x tick labels.
    axes[0, 2].tick_params(axis="x", labelbottom=True)
    legend_ax.legend(
        handles=[
            Line2D([], [], color=FRAME_COLORS[f], marker="o", ls="", ms=5,
                   label=FRAME_LABELS[f])
            for f in ("A", "D")
        ],
        loc="center", frameon=False,
    )

    fig.supxlabel("stance", fontsize=9, color=MUTED, y=0.04)
    fig.supylabel("framing", fontsize=9, color=MUTED, x=0.04)
    fig.suptitle(
        "Stance vs. framing by model",
        ha="left", fontsize=10.5, fontweight="medium", color=INK, y=1.0,
    )
    fig.text(
        0.005, -0.06,
        "+ = progressive · A audit · D casual · n=384 per model · r, SD of stance",
        **NOTE_KW,
    )
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.97))
    fig.subplots_adjust(hspace=0.25)
    align_suptitle(fig, axes[0, 0])
    save(fig, "stance_vs_framing_by_model")


def main() -> None:
    set_style()
    df = load()
    fig_persona_effect(df)
    fig_adg_by_model(df)
    fig_hedge_rate(df)
    fig_stance_framing_scatter(df)


if __name__ == "__main__":
    main()
