"""Every eval set in the repo, against the shipping criterion.

Two panels, one shared row per eval set, so the same eval set is read across both:

  LEFT   where the anchors sit on the metric. The bar is the base->reference band,
         which IS the entire scoring range -- a submission's reward is its position
         along it. `random_init` is drawn where it was measured on the same split,
         because a band that a randomly-initialised model already reaches is not a
         band.

  RIGHT  band_sigma, the band divided by seed noise, against the 4.0 bar that
         common/shipping.py derives from a 0.25 tolerance on reward noise. This is
         the panel that decides whether an eval set may ship.

Why two panels and not one: they answer different questions. A wide band can still
be unusable if the arms are noisy, and the preference track is exactly that case --
which is only visible when effect size and precision are drawn separately.

Values are read from the committed anchor files, not typed in. Run:

    research/.venv/bin/python research/plot_criterion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "common"))
from shipping import K_SCREENED, criterion_record, evaluate  # noqa: E402

# Surfaces and ink, matching research/plot_ladder.py.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BAND_FILL = "#e8e7e1"

# Categorical slots 1-3. Validated all-pairs, light mode: worst CVD dE 9.2,
# worst normal-vision dE 24.0. The green warns on contrast vs surface (2.74:1),
# so every mark it colours is also directly labelled -- that is the required
# relief, not an optional nicety.
C_RANDOM, C_BASE, C_REF = "#2a78d6", "#eb6834", "#1baf7a"

# Status slots, reserved. Always shipped with a text label, never colour alone.
GOOD, CRITICAL = "#006300", "#d03b3b"

# Verdicts live in a fixed column, right of the longest bar (15.98), so a short
# bar's label cannot collide with the threshold line or with the bar itself.
LABEL_X = 16.8

TASKS = [
    ("mol-property-adapt", "auc"),
    ("qa-sft-adapt", "acc"),
    ("pref-reward-model", "acc"),
]
# Which ladder record holds each track's random_init arm. The values themselves are
# READ from it (see random_init_means) rather than restated here: the seven means
# used to be typed into this file as a dict, under a docstring promising they were
# not, in the script the root README cites as the example of reading committed data.
# common/check_reward.py now fails on a measured value typed into a constant here.
#
# The mol track has no entry on purpose: its random-init arm was measured on the
# original scaffold split, not the shipped private one, so there is no value for
# these eval sets that came from the same data. That absence is Gate A being
# unmeasured for mol, which `common/shipping.py` now reports rather than skipping.
RANDOM_INIT_LADDERS = ("qa_anchors.json", "rm_anchors.json")


def random_init_means() -> dict[str, float]:
    """eval set -> random-init mean, from the committed ladders."""
    out: dict[str, float] = {}
    for fname in RANDOM_INIT_LADDERS:
        doc = json.loads((HERE / "posttrain" / "results" / fname).read_text())
        for name, arm in doc.get("arms", {}).get("random_init", {}).items():
            out[name] = arm["mean"]
    return out


def load_rows() -> list[dict]:
    """One row per eval set, from the committed anchors plus the ladder records."""
    rows = []
    random_init = random_init_means()
    for task, metric in TASKS:
        anchors_path = (ROOT / "tasks" / task / "tests" / "grader" / "private"
                        / "anchors.json")
        anchors = json.loads(anchors_path.read_text())
        # The preference task ships one eval set provisionally; its other three
        # were screened and rejected, and the figure is about the criterion, so
        # they belong in it. Recover them from the measured ladder.
        extra = {}
        if task == "pref-reward-model":
            led = json.loads((HERE / "posttrain" / "results" / "rm_anchors.json")
                             .read_text())
            extra = {k: v for k, v in led["screened"].items() if k not in anchors}
        for name, a in list(anchors.items()) + list(extra.items()):
            band = a.get("band", a[f"reference_{metric}"] - a[f"base_{metric}"])
            sigma = a.get("sigma") or (band / a["band_sigma"])
            v = evaluate(band=band, base_std=sigma, ref_std=sigma,
                         n_seeds=a.get("n_seeds", 5), k_screened=K_SCREENED[task])
            rows.append({
                "task": task, "name": name,
                "base": a[f"base_{metric}"], "reference": a[f"reference_{metric}"],
                "random_init": random_init.get(name),
                "band_sigma": v["band_sigma"], "noise": v["reward_noise_on_rerun"],
                "ships": v["ships"],
            })
    return rows


def main() -> None:
    rows = load_rows()
    crit = criterion_record()
    bar = crit["min_band_sigma"]

    # Group order: one block per task, separated by a blank slot.
    y, labels, blocks = [], [], []
    cursor = 0.0
    for task, _ in TASKS:
        members = [r for r in rows if r["task"] == task]
        blocks.append((task, cursor, cursor + len(members) - 1))
        for r in members:
            y.append(cursor)
            labels.append(r["name"])
            cursor += 1
        cursor += 0.8
    order = {id(r): yy for r, yy in zip(rows, y)}

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13.2, 6.4), gridspec_kw={"width_ratios": [1.55, 1.0],
                                                "wspace": 0.06})
    fig.patch.set_facecolor(SURFACE)

    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(colors=INK_2, length=0)
        a.set_axisbelow(True)

    # ---------------------------------------------------------------- panel A
    for r in rows:
        yy = order[id(r)]
        ax.plot([r["base"], r["reference"]], [yy, yy], color=BAND_FILL, lw=9,
                solid_capstyle="round", zorder=1)
        if r["random_init"] is not None:
            ax.plot(r["random_init"], yy, "o", ms=8, mfc=SURFACE, mec=C_RANDOM,
                    mew=2, zorder=3)
        ax.plot(r["base"], yy, "o", ms=9, color=C_BASE, mec=SURFACE, mew=2, zorder=4)
        ax.plot(r["reference"], yy, "o", ms=9, color=C_REF, mec=SURFACE, mew=2,
                zorder=4)
        # Direct labels: required relief for the green's sub-3:1 contrast, and
        # they carry the numbers the shapes cannot. Where the band is narrow the
        # two labels would sit almost on top of each other -- and on the
        # preference rows the band IS narrow, that being the finding -- so they
        # are pushed apart horizontally instead of being allowed to collide.
        narrow = (r["reference"] - r["base"]) < 0.04
        ax.annotate(f"{r['base']:.3f}", (r["base"], yy),
                    xytext=(-9 if narrow else 0, -14), textcoords="offset points",
                    ha="right" if narrow else "center", fontsize=7.4, color=INK_2)
        ax.annotate(f"{r['reference']:.3f}", (r["reference"], yy),
                    xytext=(9 if narrow else 0, 9), textcoords="offset points",
                    ha="left" if narrow else "center", fontsize=7.4, color=INK)

    ax.set_xlim(0.19, 0.95)
    ax.set_xlabel("metric on the private holdout  (ROC-AUC / accuracy)",
                  color=INK_2, fontsize=9)
    ax.set_title("Where the anchors sit", color=INK, fontsize=11.5,
                 loc="left", pad=26, fontweight="bold")
    ax.annotate("the shaded bar is the whole scoring range: reward is a\n"
                "submission's position from base (0) to reference (1)",
                xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", fontsize=8.6, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.7)

    # ---------------------------------------------------------------- panel B
    for r in rows:
        yy = order[id(r)]
        col = GOOD if r["ships"] else CRITICAL
        bx.plot([0, r["band_sigma"]], [yy, yy], color=col, lw=3.2,
                solid_capstyle="round", zorder=2)
        bx.plot(r["band_sigma"], yy, "o", ms=8, color=col, mec=SURFACE, mew=2,
                zorder=3)
        verdict = "ships" if r["ships"] else "fails"
        bx.text(LABEL_X, yy, f"{r['band_sigma']:>5.2f}σ   {verdict}", va="center",
                ha="left", fontsize=8.4, color=INK,
                fontweight="bold" if r["ships"] else "normal")

    bx.axvline(bar, color=INK_2, lw=1.4, ls=(0, (4, 3)), zorder=1)
    bx.annotate(f"bar: {bar:.1f}σ", xy=(bar, cursor - 0.6), xytext=(5, 0),
                textcoords="offset points", fontsize=8.4, color=INK_2,
                fontweight="bold")
    bx.set_xlim(0, 23.5)
    bx.set_xlabel("band ÷ seed noise  (band_sigma)", color=INK_2, fontsize=9)
    bx.set_title("Does the reward survive a rerun?", color=INK, fontsize=11.5,
                 loc="left", pad=26, fontweight="bold")
    bx.annotate("bar derived from a stated tolerance: a rerun of the same\n"
                "submission must not move its reward by more than 0.25",
                xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", fontsize=8.6, color=MUTED)
    bx.grid(axis="x", color=GRID, lw=0.7)

    # ------------------------------------------------------------- shared rows
    for a in (ax, bx):
        a.set_ylim(cursor - 1.0, -1.35)
        a.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    bx.set_yticklabels([])

    for task, y0, y1 in blocks:
        ax.text(-0.30, (y0 + y1) / 2, task, transform=ax.get_yaxis_transform(),
                rotation=90, va="center", ha="center", fontsize=8.4,
                color=MUTED, fontweight="bold")

    handles = [
        Line2D([], [], marker="o", ls="", ms=8, mfc=SURFACE, mec=C_RANDOM, mew=2,
               label="random-init control (same split)"),
        Line2D([], [], marker="o", ls="", ms=9, color=C_BASE,
               label="base — ceiling of everything that does not adapt"),
        Line2D([], [], marker="o", ls="", ms=9, color=C_REF,
               label="reference — a tuned, ordinary adaptation"),
    ]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.075, 0.005),
                     frameon=False, fontsize=8.8, ncol=3, columnspacing=1.6,
                     handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle("Which eval sets may be used as a reward",
                 x=0.075, y=0.975, ha="left", fontsize=14, color=INK,
                 fontweight="bold")
    fig.text(0.075, 0.932,
             "Nine eval sets across three tasks, graded by one rule "
             "(common/shipping.py). The preference track's bands are real — "
             "they are just too small to score with.",
             ha="left", fontsize=9.4, color=INK_2)
    fig.text(0.075, 0.052,
             "The mol task's random-init arm was measured on the original scaffold "
             "split rather than the private one, so it is omitted instead of shown "
             "at a value from different data.",
             ha="left", fontsize=8, color=MUTED)

    fig.subplots_adjust(left=0.175, right=0.985, top=0.845, bottom=0.135)
    dest = HERE / "results" / "shipping_criterion.png"
    fig.savefig(dest, dpi=200, facecolor=SURFACE)
    print(f"wrote {dest}")
    for r in rows:
        print(f"  {r['task']:<20}{r['name']:<14}{r['band_sigma']:>6.2f}σ  "
              f"noise {r['noise']:.2f}  {'ships' if r['ships'] else 'FAILS'}")


if __name__ == "__main__":
    main()
