"""The shipping criterion: may an eval set's reward be used?

One rule, in one place, applied to every task in this repo. It exists because the
first version of this decision was neither.

What was wrong before
---------------------
The bar was `band_sigma >= 3.0`, and 3.0 was picked by looking at what the repo
had already shipped -- `mol-property-adapt`'s bbbp at 4.09 sigma -- and going one
notch below. Calibrating a threshold against a previous decision is not deriving
it. Worse, there was a second criterion (`band >= 0.02`) that was **removed after
it excluded the eval set I wanted to keep**. The stated reason was sound (bbbp
ships on a band of 0.0143, so an absolute floor would have excluded an eval set
already considered good) but the sequence was motivated reasoning, and the
resulting 3.10-sigma pass should never have been treated as a pass.

What this replaces it with
--------------------------
A tolerance, stated up front, from which the bar is derived:

    MAX_REWARD_NOISE = 0.25

"Rerunning the same submission with a different seed must not move its reward by
more than a quarter." Since recovery is (score - base) / band, a seed wobble of
sigma becomes sigma/band of reward, so that tolerance *is* the bar:

    band >= sigma / MAX_REWARD_NOISE        i.e.  band_sigma >= 4.0

Change the tolerance and the bar moves with it, visibly. Nobody has to argue
about 3 versus 4 again; they argue about how much reward noise is acceptable,
which is the real question.

Two tests, not one
------------------
The old single test conflated two different quantities that happen to share the
word "noise":

  PRECISION  -- is the band wide compared with how much ONE submission's score
                moves between seeds? This is what decides whether the reward
                measures the submission or the seed. Uses sigma (the arms' seed
                spread) and is the test above.

  EXISTENCE  -- is the band distinguishable from zero at all? This uses the
                standard error of the *anchors* (sigma / sqrt(n_seeds)), which is
                smaller, and it is corrected for having screened several eval sets
                and shipped the best. An effect can be real and still be useless
                as a reward, which is exactly the preference track's situation.

Selection correction
--------------------
Screening k eval sets and shipping the best inflates the winner. The existence
test is Bonferroni-corrected by k, and `report()` prints k so the inflation is
never invisible.

Gate A stays
------------
An eval set also has to show the pretrained weights doing work: the reference must
beat a randomly-initialised control by more than one sigma, and significantly.
A band bought by a surface feature both arms can learn is not a band.
"""

from __future__ import annotations

import math
from statistics import NormalDist

# The one knob. Everything else follows from it.
MAX_REWARD_NOISE = 0.25

# One-sided significance for the existence and Gate-A tests, before correction.
ALPHA = 0.05


def min_band_sigma(max_reward_noise: float = MAX_REWARD_NOISE) -> float:
    """The precision bar, derived rather than chosen."""
    return 1.0 / max_reward_noise


def _z_crit(k_screened: int, alpha: float = ALPHA) -> float:
    """One-sided critical value, Bonferroni-corrected across k screened eval sets."""
    return NormalDist().inv_cdf(1.0 - alpha / max(k_screened, 1))


def evaluate(*, band: float, base_std: float, ref_std: float, n_seeds: int,
             k_screened: int, random_init_gain: float | None = None,
             random_init_std: float | None = None,
             max_reward_noise: float = MAX_REWARD_NOISE) -> dict:
    """Verdict for one eval set. Returns every intermediate so it can be audited.

    `sigma` is the conservative per-submission seed spread: the larger of the two
    arms that define the band. Using the larger rather than a pooled value means a
    noisy `base` cannot be averaged away by a tight `reference`.
    """
    sigma = max(base_std, ref_std, 1e-9)
    bar = min_band_sigma(max_reward_noise)
    band_sigma = band / sigma
    reward_noise = sigma / band if band > 0 else float("inf")

    # Existence: standard error of a difference of two means over n seeds each.
    se_band = math.sqrt((base_std ** 2 + ref_std ** 2) / max(n_seeds, 1)) or 1e-9
    z_band = band / se_band
    z_crit = _z_crit(k_screened)

    fails = []
    if band <= 0:
        fails.append(f"inverted: band {band:+.4f} is not positive")
    if band_sigma < bar:
        fails.append(
            f"imprecise: band_sigma {band_sigma:.2f} < {bar:.2f}, so a rerun of the "
            f"same submission moves the reward by {min(reward_noise, 9.99):.2f} "
            f"against a tolerance of {max_reward_noise}")
    if z_band < z_crit:
        fails.append(f"not distinguishable from zero: z {z_band:.2f} < {z_crit:.2f} "
                     f"(Bonferroni over {k_screened} screened eval sets)")

    out = {
        "band": round(band, 4),
        "sigma": round(sigma, 4),
        "band_sigma": round(band_sigma, 2),
        "reward_noise_on_rerun": round(min(reward_noise, 9.99), 3),
        "min_band_sigma": round(bar, 2),
        "band_z": round(z_band, 2),
        "z_crit_bonferroni": round(z_crit, 2),
        "k_screened": k_screened,
        "n_seeds": n_seeds,
    }

    if random_init_gain is not None:
        out["pretraining_gain"] = round(random_init_gain, 4)
        ri_std = random_init_std if random_init_std is not None else sigma
        se_gain = math.sqrt((ref_std ** 2 + ri_std ** 2) / max(n_seeds, 1)) or 1e-9
        z_gain = random_init_gain / se_gain
        out["gain_z"] = round(z_gain, 2)
        if random_init_gain < sigma:
            fails.append(
                f"pretrained weights do too little work: gain {random_init_gain:+.4f} "
                f"is under one sigma ({sigma:.4f})")
        elif z_gain < z_crit:
            fails.append(f"pretraining gain not significant: z {z_gain:.2f} < {z_crit:.2f}")

    out["ships"] = not fails
    out["failed"] = fails
    return out


def criterion_record(max_reward_noise: float = MAX_REWARD_NOISE) -> dict:
    """Stamped into every results file so the rule cannot move silently."""
    return {
        "max_reward_noise": max_reward_noise,
        "min_band_sigma": round(min_band_sigma(max_reward_noise), 2),
        "alpha_one_sided": ALPHA,
        "selection_correction": "bonferroni over screened eval sets",
        "tests": ["band positive",
                  "precision: band_sigma >= min_band_sigma",
                  "existence: band significantly > 0 after correction",
                  "gate A: reference beats random_init by >1 sigma, significantly"],
        "source": "common/shipping.py",
    }


# ------------------------------------------------------------------- reporting

# How many candidate eval sets each track screened before shipping its winners.
# Recorded here because the selection correction needs it and it is not
# recoverable from a shipped anchors.json.
K_SCREENED = {
    "mol-property-adapt": 5,      # bbbp, tox21, clintox, bace, sider (research/SPIKE_RESULTS.md)
    "qa-sft-adapt": 3,
    "pref-reward-model": 4,
    "sciml-protein-regression": 1,
}


def report() -> int:
    """Apply the criterion to every task's committed anchors.

    The point of running it over tasks that already shipped is that a bar which
    only ever rejects the newest thing is not a bar. `mol-property-adapt` predates
    this criterion and is graded by it here anyway.

    Older anchor files record only `band` and `band_sigma`, so sigma is recovered
    as band/band_sigma and both arms are assumed equally noisy. That is
    conservative for the existence test: assuming the tighter arm is as loose as
    the looser one can only inflate the standard error.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    c = criterion_record()
    print(f"shipping criterion: band_sigma >= {c['min_band_sigma']} "
          f"(derived from max_reward_noise = {c['max_reward_noise']}), "
          f"existence at alpha {c['alpha_one_sided']} Bonferroni-corrected\n")
    print(f"{'task':<26}{'eval set':<14}{'band':>8}{'b_sig':>7}{'noise':>7}"
          f"{'z':>7}  verdict")
    worst = 0
    for task in sorted(K_SCREENED):
        path = root / "tasks" / task / "tests" / "grader" / "private" / "anchors.json"
        if not path.exists():
            print(f"{task:<26}{'(no anchors — tiered or unassembled reward)':<14}")
            continue
        anchors = json.loads(path.read_text())
        for name, a in anchors.items():
            metric = next((m for m in ("auc", "acc", "spearman")
                           if f"base_{m}" in a), None)
            if metric is None:
                continue
            band = a.get("band", a[f"reference_{metric}"] - a[f"base_{metric}"])
            sigma = a.get("sigma") or (band / a["band_sigma"] if a.get("band_sigma")
                                       else None)
            if not sigma:
                print(f"{task:<26}{name:<14}  band {band:.4f}, no recorded noise")
                continue
            v = evaluate(band=band, base_std=sigma, ref_std=sigma,
                         n_seeds=a.get("n_seeds", 5),
                         k_screened=K_SCREENED[task],
                         random_init_gain=a.get("pretraining_gain"))
            print(f"{task:<26}{name:<14}{v['band']:>8.4f}{v['band_sigma']:>7.2f}"
                  f"{v['reward_noise_on_rerun']:>7.2f}{v['band_z']:>7.2f}  "
                  + ("ships" if v["ships"] else "NO -- " + v["failed"][0][:60]))
            worst += 0 if v["ships"] else 1
    print(f"\n{worst} shipped eval set(s) would not pass the current criterion")
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
