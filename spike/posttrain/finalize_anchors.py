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
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent

NO_ADAPT = {"rm": ["length_only", "frozen_probe", "frozen_head"],
            "qa": ["zero_shot", "head_only"]}
REFERENCE_ARM = {"rm": "finetune", "qa": "sft_full"}
T_FLOOR = {"rm": 0.85, "qa": 0.85}


def derive(track: str, data: dict, min_sigma: float, min_band: float) -> dict:
    cfg = data["config"]
    arms = data["arms"]
    metric = cfg.get("metric", "acc")
    ref_arm = REFERENCE_ARM[track]
    no_adapt = [a for a in NO_ADAPT[track] if a in arms]

    anchors, rejected = {}, {}
    for name in cfg["eval_sets"]:
        if ref_arm not in arms or name not in arms[ref_arm]:
            rejected[name] = f"no {ref_arm} measurement"
            continue
        ceiling_arm = max(no_adapt, key=lambda a: arms[a].get(name, {})
                          .get("mean", -1.0))
        base = arms[ceiling_arm][name]["mean"]
        ref = arms[ref_arm][name]["mean"]
        base_std = arms[ceiling_arm][name]["std"]
        ref_std = arms[ref_arm][name]["std"]
        sigma = max(base_std, ref_std, 1e-6)
        band = ref - base

        seeds = [v for a in arms.values() if name in a for v in a[name]["seeds"]]
        best_observed = max(seeds)
        t_imp = round(min(0.98, max(best_observed + 0.15, T_FLOOR[track])), 2)

        entry = {
            f"base_{metric}": round(base, 4),
            f"reference_{metric}": round(ref, 4),
            "band": round(band, 4),
            "band_sigma": round(band / sigma, 2),
            "t_implausible": t_imp,
            "best_observed": round(best_observed, 4),
            "base_arm": ceiling_arm,
            "base_definition":
                f"ceiling of the no-adaptation arms {no_adapt}: {ceiling_arm} at "
                f"{base:.4f} +/- {base_std:.4f} over "
                f"{len(arms[ceiling_arm][name]['seeds'])} seeds",
            "reference_definition":
                f"{ref_arm}, {cfg['epochs']} epochs, lr {cfg['lr']}, bs {cfg['bs']}, "
                f"best-val epoch; {len(arms[ref_arm][name]['seeds'])}-seed mean "
                f"{ref:.4f} +/- {ref_std:.4f}",
        }

        why = []
        if band <= 0:
            why.append(f"inverted: {ref_arm} ({ref:.4f}) does not beat "
                       f"{ceiling_arm} ({base:.4f})")
        elif band < min_band:
            why.append(f"band {band:.4f} < {min_band}")
        if entry["band_sigma"] < min_sigma and band > 0:
            why.append(f"band_sigma {entry['band_sigma']} < {min_sigma}")
        ri = arms.get("random_init", {}).get(name, {}).get("mean")
        if ri is not None:
            entry["random_init"] = round(ri, 4)
            entry["pretraining_gain"] = round(ref - ri, 4)
            # Gate A, and the one that is easiest to pass by accident: an eval
            # set can have a clean band between `base` and `reference` while the
            # pretrained weights contribute nothing, because both arms are
            # learning the same surface feature. The first cut of the preference
            # track did exactly that -- a randomly-initialized encoder reached
            # 0.594 against the pretrained fine-tune's 0.604, a gain smaller than
            # the seed noise. Requiring the gain to clear 2 sigma is what turns
            # "the number went up" into "the model is doing the work".
            if ref - ri < max(min_band, 2 * sigma):
                why.append(
                    f"pretrained weights do too little work: {ref_arm} {ref:.4f} "
                    f"vs random_init {ri:.4f} is +{ref - ri:.4f}, under "
                    f"max(min_band, 2 sigma) = {max(min_band, 2 * sigma):.4f}")
        if why:
            rejected[name] = "; ".join(why)
            entry["shipped"] = False
        else:
            entry["shipped"] = True
        anchors[name] = entry

    return {"anchors": {k: v for k, v in anchors.items() if v.get("shipped")},
            "screened": anchors, "rejected": rejected,
            "criteria": {"min_band_sigma": min_sigma, "min_band": min_band}}


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
    ap.add_argument("--min-sigma", type=float, default=3.0)
    # An absolute band floor is deliberately NOT the criterion. The mol task's
    # `bbbp` ships on a band of 0.0143 at 4.09 sigma, so a 0.02 floor would have
    # excluded an eval set this repo already considers good. What matters is the
    # band relative to noise; the absolute width is reported, not gated. The
    # value below is still used as the floor for the pretraining-gain check,
    # where an absolute minimum does make sense.
    ap.add_argument("--min-band", type=float, default=0.0)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    for track in (["rm", "qa"] if args.track == "both" else [args.track]):
        path = HERE / "results" / f"{track}_anchors.json"
        if not path.exists():
            print(f"SKIP {track}: {path} not measured yet")
            continue
        data = json.loads(path.read_text())
        out = derive(track, data, args.min_sigma, args.min_band)
        data.update(out)
        path.write_text(json.dumps(data, indent=2))

        metric = data["config"].get("metric", "acc")
        print(f"\n== {track}")
        print(f"{'eval set':<14}{'base':>9}{'ref':>9}{'band':>9}{'sigma':>8}"
              f"{'rand':>8}  ship")
        for name, a in out["screened"].items():
            print(f"{name:<14}{a['base_'+metric]:>9.4f}{a['reference_'+metric]:>9.4f}"
                  f"{a['band']:>9.4f}{a['band_sigma']:>8.2f}"
                  f"{a.get('random_init', float('nan')):>8.4f}"
                  f"  {'yes' if a['shipped'] else 'NO -- ' + out['rejected'][name]}")
        print(f"shipping {len(out['anchors'])} of {len(out['screened'])} eval sets")
        if args.markdown:
            print()
            print(markdown(track, data, out))


if __name__ == "__main__":
    main()
