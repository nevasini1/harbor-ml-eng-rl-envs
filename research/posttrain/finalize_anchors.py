"""Derive anchors from a measured ladder, and say which eval sets may ship.

Measurement and derivation are deliberately separate steps. `modal_measure.py`
records what every arm scored at every seed; this file turns that into the two
numbers a verifier divides by. Keeping them apart means the rule can be re-read,
argued with and re-run in a second without touching a GPU -- and it means the
anchors in a shipped task are reproducible from a committed file rather than from
a session someone ran once.

The rules, all of them:

  base       = the mean of the **best** no-adaptation arm on that eval set, not
               of a nominated one. The mol track found the frozen probe and the
               trained head swapping places between its two eval sets; pinning to
               the loser pays every lazy submission a slice of the reward for
               free. Here `head_only` beats `zero_shot` on sciq by 5 points and
               loses to it on arc_easy, so the same seam is present again.

  reference  = the mean of the tuned adaptation arm (`finetune` / `sft_full`).

  band_sigma = band / max(std of the two arms that define it). This is the number
               that decides whether an eval set ships: below it, the reward is
               scoring the seed rather than the submission.

  t_implausible = min(0.98, max(best observed + 0.15, floor)). A tripwire, not
               proof -- but derived by a stated rule from the measurements, which
               the protein task's hand-picked t_strong never was.

An eval set ships only if the ordering holds (reference > base), the pretrained
weights do work (both beat `random_init`), and band_sigma clears --min-sigma.
Everything else is reported and dropped, with the reason recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "common"))
from shipping import criterion_record, evaluate  # noqa: E402

NO_ADAPT = {"rm": ["length_only", "frozen_probe", "frozen_head"],
            "qa": ["zero_shot", "head_only"]}
REFERENCE_ARM = {"rm": "finetune", "qa": "sft_full"}
T_FLOOR = {"rm": 0.85, "qa": 0.85}


def derive(track: str, data: dict) -> dict:
    """Turn a measured ladder into anchors, and apply the shipping criterion.

    The decision is not made here. It is made by `common/shipping.py`, which every
    task shares, so an eval set cannot be admitted by a threshold that was tuned
    to admit it. This function only assembles the inputs that criterion needs.
    """
    cfg = data["config"]
    arms = data["arms"]
    metric = cfg.get("metric", "acc")
    ref_arm = REFERENCE_ARM[track]
    no_adapt = [a for a in NO_ADAPT[track] if a in arms]
    names = [n for n in cfg["eval_sets"]
             if ref_arm in arms and n in arms[ref_arm]]
    k = len(names)  # how many eval sets were screened; the winner is corrected for it

    anchors, rejected = {}, {}
    for name in names:
        ceiling_arm = max(no_adapt, key=lambda a: arms[a].get(name, {})
                          .get("mean", -1.0))
        base = arms[ceiling_arm][name]
        ref = arms[ref_arm][name]
        ri = arms.get("random_init", {}).get(name)

        verdict = evaluate(
            band=ref["mean"] - base["mean"],
            base_std=base["std"], ref_std=ref["std"],
            n_seeds=len(ref["seeds"]), k_screened=k,
            random_init_gain=(ref["mean"] - ri["mean"]) if ri else None,
            random_init_std=ri["std"] if ri else None)

        seeds = [v for a in arms.values() if name in a for v in a[name]["seeds"]]
        best_observed = max(seeds)

        entry = {
            f"base_{metric}": round(base["mean"], 4),
            f"reference_{metric}": round(ref["mean"], 4),
            "t_implausible": round(min(0.98, max(best_observed + 0.15,
                                                 T_FLOOR[track])), 2),
            "best_observed": round(best_observed, 4),
            "base_arm": ceiling_arm,
            "base_definition":
                f"ceiling of the no-adaptation arms {no_adapt}: {ceiling_arm} at "
                f"{base['mean']:.4f} +/- {base['std']:.4f} over {len(base['seeds'])} seeds",
            "reference_definition":
                f"{ref_arm}, {cfg['epochs']} epochs, lr {cfg['lr']}, bs {cfg['bs']}, "
                f"best-val epoch; {len(ref['seeds'])}-seed mean {ref['mean']:.4f} "
                f"+/- {ref['std']:.4f}",
            **verdict,
        }
        if ri:
            entry["random_init"] = round(ri["mean"], 4)
        entry["shipped"] = verdict["ships"]
        if not verdict["ships"]:
            rejected[name] = "; ".join(verdict["failed"])
        anchors[name] = entry

    return {"anchors": {k2: v for k2, v in anchors.items() if v["shipped"]},
            "screened": anchors, "rejected": rejected,
            "criterion": criterion_record()}


def markdown(track: str, data: dict, out: dict) -> str:
    """The ladder and the anchors as tables, so RESULTS.md is transcribed by a
    machine rather than by hand."""
    cfg, arms = data["config"], data["arms"]
    metric = cfg.get("metric", "acc")
    names = cfg["eval_sets"]
    lines = [f"| arm | {' | '.join(names)} |",
             f"|---|{'---|' * len(names)}"]
    for arm in cfg["arms"]:
        cells = []
        for n in names:
            a = arms.get(arm, {}).get(n)
            cells.append(f"{a['mean']:.4f} ± {a['std']:.4f}" if a else "—")
        lines.append(f"| `{arm}` | {' | '.join(cells)} |")
    lines += ["", "| eval set | base | reference | band | band σ | pretraining gain | ships |",
              "|---|---|---|---|---|---|---|"]
    for name, a in out["screened"].items():
        gain = a.get("pretraining_gain")
        lines.append(
            f"| `{name}` | {a['base_' + metric]:.4f} ({a['base_arm']}) | "
            f"{a['reference_' + metric]:.4f} | {a['band']:.4f} | "
            f"**{a['band_sigma']:.2f}σ** | "
            f"{gain:+.4f} | " + ("**yes**" if a["shipped"]
                                 else f"no — {out['rejected'][name]}") + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["rm", "qa", "both"], default="both")


    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    for track in (["rm", "qa"] if args.track == "both" else [args.track]):
        path = HERE / "results" / f"{track}_anchors.json"
        if not path.exists():
            print(f"SKIP {track}: {path} not measured yet")
            continue
        data = json.loads(path.read_text())
        out = derive(track, data)
        # Clear last run's derived keys before merging this run's in. `update`
        # merges, so anything the previous pipeline wrote and this one no longer
        # writes survives forever: both files carried a `criteria` block from a
        # pre-4.0 run -- min_band_sigma 3.0 and the `min_band` test that
        # common/shipping.py records as removed -- sitting one character away from
        # the live `criterion` key, read by nothing. The measurement (`config`,
        # `arms`) is what must be preserved across a re-derivation; the verdict
        # must not be.
        for stale in ("anchors", "screened", "rejected", "criterion", "criteria"):
            data.pop(stale, None)
        data.update(out)
        path.write_text(json.dumps(data, indent=2))

        metric = data["config"].get("metric", "acc")
        c = out["criterion"]
        print(f"\n== {track}   bar: band_sigma >= {c['min_band_sigma']} "
              f"(from max_reward_noise {c['max_reward_noise']})")
        print(f"{'eval set':<14}{'base':>9}{'ref':>9}{'band':>9}{'sigma':>7}"
              f"{'noise':>8}{'z':>7}  ship")
        for name, a in out["screened"].items():
            print(f"{name:<14}{a['base_'+metric]:>9.4f}{a['reference_'+metric]:>9.4f}"
                  f"{a['band']:>9.4f}{a['band_sigma']:>7.2f}"
                  f"{a['reward_noise_on_rerun']:>8.2f}{a['band_z']:>7.2f}  "
                  + ("yes" if a["shipped"] else "NO -- " + out["rejected"][name]))
        print(f"shipping {len(out['anchors'])} of {len(out['screened'])} eval sets")
        if args.markdown:
            print()
            print(markdown(track, data, out))


if __name__ == "__main__":
    main()
