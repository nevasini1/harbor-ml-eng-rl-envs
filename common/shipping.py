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

    Absent noise is a refusal, not a floor. This used to read
    `max(base_std, ref_std, 1e-9)`, which turned "no seed spread was recorded" into
    `band_sigma` of order 1e7 and a rerun noise of ~0 -- a malformed measurement
    passed the precision and existence tests by seven orders of magnitude and read
    as the most precise eval set in the repo. One arm being deterministic is
    legitimate and still works, because `max` takes the other one; both being zero
    means nothing was measured. `verifier_core.load_anchors` states the policy for
    the whole package: "There is deliberately no fallback. A silent default would
    regrade the whole field against a bar nobody chose while still reporting status
    ok."
    """
    if base_std <= 0.0 and ref_std <= 0.0:
        raise ValueError(
            f"no seed spread recorded for either arm (base_std={base_std}, "
            f"ref_std={ref_std}); band_sigma divides by this, so a missing "
            "measurement cannot be defaulted. Re-measure, or pass the arm's real "
            "standard deviation.")
    sigma = max(base_std, ref_std)
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
        out["gate_a"] = "passed"
        if random_init_gain < sigma:
            out["gate_a"] = "failed"
            fails.append(
                f"pretrained weights do too little work: gain {random_init_gain:+.4f} "
                f"is under one sigma ({sigma:.4f})")
        elif z_gain < z_crit:
            out["gate_a"] = "failed"
            fails.append(f"pretraining gain not significant: z {z_gain:.2f} < {z_crit:.2f}")
    else:
        # Not measured is its own state, and it is recorded rather than inferred.
        # This branch used to be absent entirely, so an eval set with no
        # random-init arm ran three of the four tests `criterion_record()`
        # advertises and still printed a bare "ships". "We could not check this"
        # and "this passed" are different claims and the output now distinguishes
        # them. It is deliberately NOT a failure: mol's random-init arm exists but
        # was measured on a superseded split, and refusing to assemble over a
        # missing measurement would take a shipped task offline rather than
        # describe it accurately. `ships_caveats` is what carries it forward.
        out["gate_a"] = "not measured"

    out["ships"] = not fails
    out["ships_caveats"] = ([] if out["gate_a"] != "not measured" else
                            ["gate A not measured: no random-init control on this "
                             "split, so whether the pretrained weights do the work "
                             "is unverified rather than confirmed"])
    out["failed"] = fails
    return out


# The rule's own identity. `criterion_record()` was already "stamped into every
# results file so the rule cannot move silently" -- but it recorded the rule's
# *parameters*, not its version, so an anchor screened under the retired bar and one
# screened under the current bar were distinguishable only by dating the commit that
# wrote them. Bump RULE_VERSION whenever a test is added, removed, or redefined, and
# append the outgoing rule to SUPERSEDED. Changing MAX_REWARD_NOISE alone is not a
# version bump: the whole point is that the bar follows the tolerance, and the
# tolerance is already recorded.
RULE_ID = "shipping/band-sigma-from-reward-noise"
RULE_VERSION = 2

# Retired rules, newest last. Kept rather than deleted: it is still relevant to
# know that a number was once judged this way, which is the whole reason an anchor
# needs a rule version at all.
SUPERSEDED = (
    {
        "rule_version": 1,
        "min_band_sigma": 3.0,
        "tests": ["band positive",
                  "precision: band_sigma >= 3.0",
                  "absolute floor: band >= 0.02"],
        "retired_because":
            "3.0 was picked by going one notch below what the repo had already "
            "shipped, which calibrates a threshold against a previous decision "
            "rather than deriving it; and the `band >= 0.02` floor was removed "
            "after it excluded an eval set the author wanted to keep, which is "
            "motivated reasoning. Replaced by a bar derived from a stated "
            "tolerance on reward noise.",
    },
)


def criterion_record(max_reward_noise: float = MAX_REWARD_NOISE) -> dict:
    """Stamped into every results file so the rule cannot move silently."""
    return {
        "rule_id": RULE_ID,
        "rule_version": RULE_VERSION,
        "max_reward_noise": max_reward_noise,
        "min_band_sigma": round(min_band_sigma(max_reward_noise), 2),
        "alpha_one_sided": ALPHA,
        "selection_correction": "bonferroni over screened eval sets",
        "tests": ["band positive",
                  "precision: band_sigma >= min_band_sigma",
                  "existence: band significantly > 0 after correction",
                  "gate A: reference beats random_init by >1 sigma, significantly"],
        "sigma_definition": "max(base_std, ref_std) over the two arms defining the band",
        "supersedes": [r["rule_version"] for r in SUPERSEDED],
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
    worst = provisional = caveated = 0
    seen: set[str] = set()
    for task in sorted(K_SCREENED):
        path = root / "tasks" / task / "tests" / "grader" / "private" / "anchors.json"
        if not path.exists():
            print(f"{task:<26}{'(no anchors — tiered or unassembled reward)':<14}")
            continue
        anchors = json.loads(path.read_text())
        seen.add(task)
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
                         k_screened=a.get("k_screened", K_SCREENED[task]),
                         random_init_gain=a.get("pretraining_gain"))
            # Prefer what the assembler recorded. It had the true per-arm noise;
            # this reconstruction has to assume both arms are as loose as the
            # looser one, which understates z. Printing the reconstruction next to
            # a file that already states the real number is two answers to one
            # question -- the thing this criterion exists to stop.
            v = {**v, **{k: a[k] for k in
                         ("band_sigma", "reward_noise_on_rerun", "band_z")
                         if k in a}}
            if v["ships"]:
                verdict = "ships"
                if v.get("gate_a") == "not measured":
                    verdict = "ships (gate A NOT MEASURED)"
                    caveated += 1
            elif a.get("provisional"):
                # Acknowledged in the shipped file itself. Still printed as a
                # failure, but it does not fail the build -- the same allowance
                # `--allow-provisional` grants at assembly time.
                verdict = "NO (provisional, acknowledged) -- " + v["failed"][0][:40]
                provisional += 1
            else:
                verdict = "NO -- " + v["failed"][0][:60]
                worst += 1
            print(f"{task:<26}{name:<14}{v['band']:>8.4f}{v['band_sigma']:>7.2f}"
                  f"{v['reward_noise_on_rerun']:>7.2f}{v['band_z']:>7.2f}  " + verdict)

    if not seen:
        print("\nno anchors found for any task -- this report checked nothing")
        return 1

    print(f"\n{worst} unacknowledged failure(s), {provisional} acknowledged as "
          f"provisional, {caveated} shipping with gate A unmeasured, "
          f"{len(seen)} eval set(s) examined")
    for task in sorted(K_SCREENED):
        if task not in seen:
            print(f"  NOT EXAMINED: {task} -- no anchors.json. A tiered or "
                  "unassembled reward is exempt from this criterion, which is a "
                  "gap in the criterion rather than a property of the task.")
    # Exit non-zero only for failures nobody has acknowledged. This used to be an
    # unconditional `return 0`, so the report printed "1 shipped eval set would not
    # pass" and exited green -- the one rule whose job is to reject an eval set had
    # no enforcement surface anywhere, including in CI.
    return 1 if worst else 0


if __name__ == "__main__":
    raise SystemExit(report())
