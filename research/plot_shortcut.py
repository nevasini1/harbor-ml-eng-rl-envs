"""How a zero-parameter heuristic beat a fine-tune, and what removing it changed.

The preference track's first cut measured response length instead of preference.
This figure is that finding and its repair, in the two quantities that matter:

  LEFT   "pick the longer response" -- no model, no parameters -- against the arms
         it was supposed to lose to. On the natural sample it beat the frozen
         ceiling on every eval set and drew with a full fine-tune on helpful_base.
         After balancing it reads exactly 0.5000, by construction.

  RIGHT  Gate A: what the pretrained weights were worth, measured as
         reference - random_init. On the natural sample it was +0.010 on
         helpful_base, smaller than the seed noise, i.e. nothing. Balancing the
         split at the SAME training size raises it to +0.061; more data then takes
         it to +0.082.

The three states are separated deliberately so the two changes are not confounded:
balancing and adding data were done one at a time, and the middle column is what
isolates balancing.

Sources, all committed:
  posttrain/results/rm_length_shortcut.json     the longer-wins rates
  posttrain/results/rm_ladder_unbalanced.json   natural sample, 8k train
  posttrain/results/rm_ladder_balanced_8k.json  balanced, same 8k train
  posttrain/results/rm_anchors.json             balanced, 38k train (shipped split)

Run: research/.venv/bin/python research/plot_shortcut.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).parent
RES = HERE / "posttrain" / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
CHANCE = "#c3c2b7"

# Categorical slots 1-3, validated all-pairs in light mode (worst CVD dE 9.2,
# worst normal-vision dE 24.0). Slot 3 warns at 2.74:1 against the surface, so
# every mark it colours is directly labelled -- the required relief.
C_NATURAL, C_BAL8K, C_BAL38K = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"

EVAL_SETS = ["helpful_base", "helpful_rs", "harmless", "online"]
STATES = [
    ("natural sample, 8k train", "rm_ladder_unbalanced.json", C_NATURAL),
    ("length-balanced, 8k train", "rm_ladder_balanced_8k.json", C_BAL8K),
    ("length-balanced, 38k train", "rm_anchors.json", C_BAL38K),
]


def load():
    shortcut = json.loads((RES / "rm_length_shortcut.json").read_text())
    states = []
    for label, fname, colour in STATES:
        arms = json.loads((RES / fname).read_text())["arms"]
        states.append({"label": label, "colour": colour, "arms": arms})
    return shortcut, states


def main() -> None:
    shortcut, states = load()
    y = {name: i for i, name in enumerate(EVAL_SETS)}

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(12.6, 5.6), gridspec_kw={"width_ratios": [1.0, 1.0],
                                                "wspace": 0.30})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(colors=INK_2, length=0)
        a.set_axisbelow(True)
        a.grid(axis="x", color=GRID, lw=0.7)
        a.set_ylim(len(EVAL_SETS) - 0.45, -0.75)
        a.set_yticks(list(y.values()))

    # -------------------------------------------------- panel A: the shortcut
    ax.axvline(0.5, color=CHANCE, lw=1.4, zorder=1)
    ax.annotate("chance", xy=(0.5, -0.62), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=MUTED)
    for name in EVAL_SETS:
        yy = y[name]
        nat = shortcut["unbalanced"][name]["longer_wins"]
        bal = shortcut["balanced"][name]["longer_wins"]
        ft = states[0]["arms"]["finetune"][name]["mean"]

        ax.plot([bal, nat], [yy, yy], color=GRID, lw=2.4, zorder=1,
                solid_capstyle="round")
        # The fine-tune it was supposed to lose to, on the same natural sample.
        ax.plot(ft, yy, "|", ms=15, mew=2.2, color=CRITICAL, zorder=4)
        ax.plot(nat, yy, "o", ms=9, color=C_NATURAL, mec=SURFACE, mew=1.6, zorder=3)
        ax.plot(bal, yy, "o", ms=9, color=C_BAL38K, mec=SURFACE, mew=1.6, zorder=3)
        ax.annotate(f"{nat:.4f}", (nat, yy), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=7.8, color=INK)
        ax.annotate("0.5000", (bal, yy), xytext=(0, -15),
                    textcoords="offset points", ha="center", fontsize=7.8,
                    color=INK_2)

    ax.set_xlim(0.38, 0.66)
    ax.set_xlabel("accuracy of “pick the longer response”", color=INK_2, fontsize=9)
    ax.set_title("A heuristic with no parameters", color=INK, fontsize=11.5,
                 loc="left", pad=30, fontweight="bold")
    ax.annotate("red tick = a full fine-tune on the same natural sample.\n"
                "On helpful_base the heuristic drew with it: 0.6031 vs 0.6042.",
                xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_yticklabels(EVAL_SETS, color=INK, fontsize=9.5)

    # ------------------------------------------- panel B: what pretraining bought
    bx.axvline(0.0, color=CHANCE, lw=1.4, zorder=1)
    for name in EVAL_SETS:
        yy = y[name]
        pts = []
        for st in states:
            arms = st["arms"]
            if "finetune" not in arms or "random_init" not in arms:
                continue
            gain = arms["finetune"][name]["mean"] - arms["random_init"][name]["mean"]
            sigma = max(arms["finetune"][name]["std"],
                        arms["random_init"][name]["std"])
            pts.append((gain, sigma, st["colour"]))
        if len(pts) > 1:
            bx.plot([p[0] for p in pts], [yy] * len(pts), color=GRID, lw=2.4,
                    zorder=1, solid_capstyle="round")
        # Where two states land at nearly the same gain -- `online` is +0.042 and
        # +0.046 -- vertical alternation alone puts their labels on top of each
        # other, so near-ties are also pushed apart horizontally.
        xs = [pt[0] for pt in pts]
        tie = len(xs) > 2 and abs(xs[0] - xs[-1]) < 0.02
        for i, (gain, sigma, colour) in enumerate(pts):
            # One sigma of seed noise, drawn true to scale: a gain inside its own
            # noise bar is not a gain, and that is the whole point of the first
            # column.
            bx.plot([gain - sigma, gain + sigma], [yy, yy], color=colour, lw=1.2,
                    alpha=0.55, zorder=2, solid_capstyle="butt")
            bx.plot(gain, yy, "o", ms=8.5, color=colour, mec=SURFACE, mew=1.6,
                    zorder=3)
            # Every state is labelled, not just the ends: on `harmless` the gain
            # goes 0.153 -> 0.098 -> 0.174, and labelling only the ends would hide
            # that balancing pushed it DOWN there.
            dx, ha = 0, "center"
            if tie and i == 0:
                dx, ha = -7, "right"
            elif tie and i == len(pts) - 1:
                dx, ha = 7, "left"
            bx.annotate(f"{gain:+.3f}", (gain, yy),
                        xytext=(dx, 11 if i % 2 == 0 else -16),
                        textcoords="offset points", ha=ha, fontsize=7.6,
                        color=INK if i != 1 else INK_2)

    bx.set_xlim(-0.035, 0.225)
    bx.set_xlabel("reference − random-init  (what the pretrained weights are worth)",
                  color=INK_2, fontsize=9)
    bx.set_title("Gate A, before and after", color=INK, fontsize=11.5,
                 loc="left", pad=30, fontweight="bold")
    bx.annotate("thin line through each dot is ±1σ of seed noise. On helpful_base\n"
                "and helpful_rs the natural-sample gain sits inside it — i.e. nothing.",
                xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", fontsize=8.5, color=MUTED)
    bx.set_yticklabels([])

    handles = [Line2D([], [], marker="o", ls="", ms=8.5, color=c, label=l)
               for l, _, c in STATES]
    handles.append(Line2D([], [], marker="|", ls="", ms=13, mew=2.2,
                          color=CRITICAL, label="full fine-tune (natural sample)"))
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.062, -0.004),
                     frameon=False, fontsize=8.7, ncol=4, columnspacing=1.5,
                     handletextpad=0.5)
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle("The preference task measured response length, not preference",
                 x=0.062, y=0.975, ha="left", fontsize=14, color=INK,
                 fontweight="bold")
    fig.text(0.062, 0.910,
             "Balancing the split and adding data were done one at a time, so the "
             "middle state is what isolates the effect of balancing.",
             ha="left", fontsize=9.3, color=INK_2)
    fig.text(0.062, 0.072,
             "harmless is the exception worth reading: its natural-sample gain of "
             "+0.153 looks healthy but is itself a length artifact — length is "
             "anti-correlated there (0.4116), so the\nrandom-init control scored "
             "0.4290, far below chance, by learning the rule with the wrong sign. "
             "A large Gate-A gap can be produced by a shortcut as easily as removed by one.",
             ha="left", fontsize=7.9, color=MUTED)

    fig.subplots_adjust(left=0.115, right=0.985, top=0.745, bottom=0.215)
    dest = HERE / "results" / "length_shortcut.png"
    fig.savefig(dest, dpi=200, facecolor=SURFACE)
    print(f"wrote {dest}")
    for name in EVAL_SETS:
        nat = shortcut["unbalanced"][name]["longer_wins"]
        gains = []
        for st in states:
            a = st["arms"]
            if "random_init" in a:
                gains.append(a["finetune"][name]["mean"] - a["random_init"][name]["mean"])
        print(f"  {name:<14} longer-wins {nat:.4f} -> 0.5000   "
              f"gate A gain " + " -> ".join(f"{g:+.4f}" for g in gains))


if __name__ == "__main__":
    main()
