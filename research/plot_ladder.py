"""Effort-ladder chart for the three eval sets.

Every panel spans exactly 0.20 metric units, so a horizontal distance means the
same thing in all three -- the x-ranges are offset, not rescaled.

Error bars are drawn true to scale, which makes most of them sub-pixel at this
width. That is itself the finding, so the magnitude is not faked: each dot
carries its own +/-1 sd and, below it, that sd as a share of the band the reward
has to resolve. Both numbers sit at the dot they describe.

Sources:
  tox21, bbbp   tasks/sciml-protein-regression/scripts/mol_headroom{,_bbbp}.json  (n=5)
  bbbp          scripts/bbbp_split_v2.json (re-split, anchor selected on separation)
  meltome       commit ef5f524 (shipped split), tests/tiers.json
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND = "#efeee9"
CRITICAL = "#d03b3b"        # status slot, always shipped with a text label
GOOD = "#006300"

# Categorical slots 1-3, validated all-pairs light mode.
C_PROBE, C_HEAD, C_FT = "#2a78d6", "#eb6834", "#1baf7a"

RUNGS = ["frozen probe\n(logreg / ridge)", "frozen backbone\n+ trained head", "fine-tune"]
COLORS = [C_PROBE, C_HEAD, C_FT]

NOISE_LIMIT = 10.0  # % of band; above this, rerunning the same submission regrades it

PANELS = [
    dict(
        title="tox21  ·  ROC-AUC",
        verdict="Ordered and separated",
        detail="fine-tune +0.068 over frozen head (6.5$\\sigma$)   ·   band = 0.0678",
        xlim=(0.56, 0.76),
        vals=[0.5822, 0.6341, 0.7019],
        errs=[0.0, 0.0089, 0.0055],
        noise=[None, 13.1, 8.1],
        band=(0.6341, 0.7019),
        band_label="reward band\n(base $\\rightarrow$ reference)",
        marks=[],
        note="band edges land on the two dots it\nmust separate — this is the target shape",
    ),
    dict(
        title="meltome (protein)  ·  Spearman",
        verdict="Inverted — unfixable",
        detail="fine-tune $-$0.028 BELOW frozen head ($-$7.5$\\sigma$)   ·   band = 0.0613",
        xlim=(0.37, 0.57),
        vals=[0.3887, 0.5494, 0.5214],
        errs=[0.0, 0.0033, 0.0017],
        noise=[None, 5.4, 2.8],
        band=(0.3887, 0.45),
        band_label="reward band\n($t_{weak} \\rightarrow t_{strong}$)",
        marks=[],
        note="both frozen and fine-tune clear the top tier;\nnoise is fine — the ordering is what kills it",
    ),
    dict(
        title="bbbp  ·  ROC-AUC  (re-split)",
        verdict="Ordered and separated",
        detail="fine-tune +0.014 over best frozen (4.1$\\sigma$)   ·   band = 0.0143",
        xlim=(0.80, 1.00),
        vals=[0.8978, 0.8934, 0.9121],
        errs=[0.0, 0.0028, 0.0021],
        noise=[None, 19.6, 14.7],
        band=(0.8978, 0.9121),
        band_label="reward band\n(base $\\rightarrow$ reference)",
        marks=[],
        note="the probe beats the trained head here, so base is\nthe probe: the ceiling of both, not one of them",
    ),
]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.4), dpi=200)
fig.subplots_adjust(left=0.135, right=0.985, top=0.735, bottom=0.185, wspace=0.20)

y = [0, 1, 2]

for ax, p in zip(axes, PANELS):
    if p["band"]:
        ax.axvspan(p["band"][0], p["band"][1], color=BAND, zorder=0, lw=0)
        for xv in p["band"]:
            ax.axvline(xv, color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(
            (p["band"][0] + p["band"][1]) / 2, 2.72, p["band_label"],
            ha="center", va="center", fontsize=7.6, color=MUTED, linespacing=1.5,
        )

    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=1)

    for yi, val, err, pct, col in zip(y, p["vals"], p["errs"], p["noise"], COLORS):
        if err > 0:
            ax.errorbar(val, yi, xerr=err, color=col, lw=2.0, capsize=4,
                        capthick=2.0, zorder=3)
        ax.plot([val], [yi], "o", ms=9, color=col, mec=SURFACE, mew=2.0, zorder=4)

        ax.annotate(f"{val:.4f}   ±{err:.4f}" if err > 0 else f"{val:.4f}   ±0",
                    (val, yi), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=9.0, color=INK_2, fontweight="medium")

        if pct is None:
            sub, subcol = "deterministic — no seed noise", MUTED
        elif pct > NOISE_LIMIT:
            sub, subcol = f"{pct:.0f}% of band — over limit", CRITICAL
        else:
            sub, subcol = f"{pct:.0f}% of band — within limit", GOOD
        # Keep the sub-label inside the panel: it is ~36% of the panel width, so
        # its centre has to stay clear of both edges even when the dot is not.
        lo, hi = p["xlim"]
        cx = min(max(val, lo + 0.20 * (hi - lo)), hi - 0.20 * (hi - lo))
        ax.annotate(sub, (cx, yi), textcoords="offset points", xytext=(0, -24),
                    ha="center", fontsize=8.2, color=subcol)

    for xv, lab, col in p["marks"]:
        ax.axvline(xv, color=col, lw=1.6, zorder=2)
        ax.text(xv, -0.85, lab, ha="center", va="center", fontsize=8.2,
                color=col, fontweight="bold")

    ax.set_xlim(*p["xlim"])
    ax.set_ylim(-1.05, 3.1)
    ax.set_yticks(y)
    ax.set_yticklabels(RUNGS if ax is axes[0] else [])
    ax.tick_params(axis="y", length=0, labelsize=9.4, colors=INK_2)
    ax.tick_params(axis="x", length=0, labelsize=8.6, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.text(0.0, 1.215, p["title"], transform=ax.transAxes, fontsize=11.2,
            color=INK, fontweight="bold", ha="left", va="baseline")
    ax.text(0.0, 1.122, p["verdict"], transform=ax.transAxes, fontsize=9.6,
            color=INK, ha="left", va="baseline")
    ax.text(0.0, 1.042, p["detail"], transform=ax.transAxes, fontsize=8.8,
            color=MUTED, ha="left", va="baseline")
    ax.text(0.982, 0.025, p["note"], transform=ax.transAxes, fontsize=7.8,
            color=MUTED, ha="right", va="bottom", linespacing=1.7, style="italic")

fig.suptitle(
    "Measure the whole ladder, not two anchors",
    x=0.0135, y=0.965, ha="left", fontsize=15.5, color=INK, fontweight="bold",
)
fig.text(
    0.0135, 0.925,
    "The band's left edge belongs on the highest rung that does not adapt the encoder; its right edge on the rung that does. "
    "It must also be wide relative to seed noise.\n"
    "tox21 and bbbp are the on-contract measurements that now ship; meltome is shown as measured when it was diagnosed and is not part of the task.",
    ha="left", va="top", fontsize=9.2, color=INK_2, linespacing=1.6,
)
fig.text(
    0.0135, 0.038,
    "Every panel spans exactly 0.20 metric units, so equal horizontal distance means an equal gap — x-ranges are offset, not rescaled. Error bars are ±1 sd over 5 seeds,\n"
    "drawn true to scale, so where no bar is visible the sd is genuinely sub-pixel. The % beneath each dot restates that sd against the band — the scale that decides if it matters.",
    ha="left", fontsize=8.2, color=MUTED, linespacing=1.6,
)

handles = [
    Line2D([], [], marker="o", ls="", ms=8, color=c, mec=SURFACE, mew=1.5, label=l)
    for c, l in zip(COLORS, ["frozen probe", "frozen + trained head", "fine-tune"])
]
fig.legend(
    handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.022), ncol=3,
    frameon=False, fontsize=8.8, labelcolor=INK_2, handletextpad=0.5,
    columnspacing=1.8,
)

out = "research/results/anchor_ladder.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
