"""Effort-ladder chart for the three eval sets.

Every panel spans exactly 0.20 metric units, so a horizontal distance means the
same thing in all three -- the x-ranges are offset, not rescaled.

Error bars are drawn true to scale, which makes most of them sub-pixel at this
width. That is itself the finding, so the magnitude is not faked: each dot
carries its own +/-1 sd and, below it, that sd as a share of the band the reward
has to resolve. Both numbers sit at the dot they describe.

Every number below is read from a committed file. It used to be typed in here,
and the figure went stale the moment a measurement moved: the meltome panel
carried the verdict "Inverted -- unfixable" for two days after
`tasks/sciml-protein-regression/README.md` had overturned it on the strength of
`scripts/lpft.json`, while the rendered PNG stayed embedded in the mol task's
README. A figure that restates numbers cannot be re-derived, so it silently
becomes the oldest claim in the repo. `common/check_reward.py` now fails if any
plot script hardcodes a value a committed anchor owns.

Sources:
  tox21, bbbp   research/results/anchors_private.json  (measured_arms_n5_legal, n=5)
                tasks/mol-property-adapt/tests/grader/private/anchors.json (the
                shipped base/reference, so the band drawn is the band that scores)
  meltome       tasks/sciml-protein-regression/scripts/lpft.json  (n=5, on-contract
                CLS frozen head vs LP-FT) and tests/tiers.json (the tiers that ship)
"""

import json
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent.parent


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


MOL_MEASURED = load("research/results/anchors_private.json")
MOL_SHIPPED = load("tasks/mol-property-adapt/tests/grader/private/anchors.json")
LPFT = load("tasks/sciml-protein-regression/scripts/lpft.json")
TIERS = load("tasks/sciml-protein-regression/tests/tiers.json")

# The three rungs, in the order they are drawn, and the key each is stored under.
MOL_ARMS = ("frozen_logreg_cls", "frozen_head_cls", "finetune_cls")


def mol_panel(name: str, title: str, xlim: tuple[float, float], note: str) -> dict:
    """A mol panel, with the band taken from the file the verifier divides by.

    The base arm is not nominated here: it is recovered by matching the shipped
    `base_auc` against the measured arms. That is the "base is the ceiling of the
    trivial class" rule made visible -- tox21's ceiling is the trained head and
    bbbp's is the logistic probe, and the caption follows the measurement rather
    than being written to agree with it.
    """
    measured = MOL_MEASURED[name]
    shipped = MOL_SHIPPED[name]
    arms = measured["measured_arms_n5_legal"]
    base, ref = shipped["base_auc"], shipped["reference_auc"]
    pct = measured["noise_as_pct_of_band"]

    base_arm = min(MOL_ARMS, key=lambda a: abs(arms[a]["mean"] - base))
    base_label = "frozen head" if base_arm == "frozen_head_cls" else "best frozen"

    return dict(
        title=title,
        verdict="Ordered and separated",
        detail=f"fine-tune +{ref - base:.3f} over {base_label} "
               f"({shipped['band_sigma']:.1f}$\\sigma$)   ·   band = {ref - base:.4f}",
        xlim=xlim,
        vals=[arms[a]["mean"] for a in MOL_ARMS],
        errs=[arms[a].get("std", 0.0) for a in MOL_ARMS],
        noise=[None, pct["frozen_head"], pct["finetune"]],
        band=(base, ref),
        band_label="reward band\n(base $\\rightarrow$ reference)",
        marks=[],
        note=note,
    )


def meltome_panel() -> dict:
    """The protein panel, re-measured on contract.

    What this panel used to say -- "Inverted -- unfixable" -- was drawn from an
    arm that unfroze two layers from a randomly initialised head, and against a
    frozen ceiling that used mean pooling the submission contract cannot express.
    `lpft.json` re-measured both on contract and the ordering holds. What is
    actually broken is the reward: `t_weak` and `t_strong` are drawn as marks
    here precisely because they both sit *below* the frozen ceiling, so a
    submission that never touches the encoder also scores 1.0.
    """
    frozen, lpft = LPFT["frozen"], LPFT["lpft"]
    pct = LPFT["noise_as_pct_of_band"]
    return dict(
        title="meltome (protein)  ·  Spearman",
        verdict="Ordered — but the shipped tiers miss the band",
        detail=f"LP-FT +{lpft['mean'] - frozen['mean']:.3f} over frozen head "
               f"({LPFT['band_sigma']:.2f}$\\sigma$)   ·   band = {LPFT['band']:.4f}",
        xlim=(0.375, 0.575),
        vals=[TIERS["t_weak"], frozen["mean"], lpft["mean"]],
        errs=[0.0, frozen["std"], lpft["std"]],
        noise=[None, pct["frozen"], pct["lpft"]],
        band=(frozen["mean"], lpft["mean"]),
        band_label="measured band\n(frozen $\\rightarrow$ LP-FT)",
        marks=[(TIERS["t_weak"], "$t_{weak}$", CRITICAL),
               (TIERS["t_strong"], "$t_{strong}$", CRITICAL)],
        note="both shipped tiers sit below the frozen ceiling,\n"
             "so a submission that never adapts also scores 1.0",
    )


PANELS = [
    mol_panel("tox21", "tox21  ·  ROC-AUC", (0.56, 0.76),
              "band edges land on the two dots it\nmust separate — this is the target shape"),
    meltome_panel(),
    mol_panel("bbbp", "bbbp  ·  ROC-AUC  (re-split)", (0.80, 1.00),
              "the probe beats the trained head here, so base is\n"
              "the probe: the ceiling of both, not one of them"),
]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.4), dpi=200)
# top leaves room for three stacked labels above each panel (title, verdict,
# detail) plus the two-line figure subtitle above those, which used to collide.
fig.subplots_adjust(left=0.135, right=0.985, top=0.705, bottom=0.185, wspace=0.20)

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

    # Threshold marks are labelled at the top, level with the band label: they
    # sit far to the left of the band by construction (that is the finding), so
    # the two never collide, and the bottom of the panel stays clear for the note.
    for xv, lab, col in p["marks"]:
        ax.axvline(xv, color=col, lw=1.6, zorder=2)
        ax.text(xv, 2.72, lab, ha="center", va="center", fontsize=8.2,
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
    "tox21 and bbbp ship this band as their reward. meltome's is measured but not shipped: its reward is three fixed tiers, and both of them fall short of the band's left edge.",
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
