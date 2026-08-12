---
id: 0006
title: Shipping bar derived from a stated reward-noise tolerance
status: accepted
date: 2026-08-10
supersedes: [0005]
superseded_by: null
affects: [mol-property-adapt, qa-sft-adapt, pref-reward-model, sciml-protein-regression]
commits: [56c009e, 508cd60]
rule_version: 2
---

## Context

0005 chose a threshold and then removed the test that excluded a wanted eval set.
The problem was not the numbers but that the decision had no derivation, so it could
be argued either way and moved when inconvenient.

## Decision

State the tolerance up front and let the bar follow from it:

```
MAX_REWARD_NOISE = 0.25      ->      band_sigma >= 4.0
```

"Rerunning the same submission with a different seed must not move its reward by
more than a quarter." Since recovery is `(score - base) / band`, a seed wobble of
sigma becomes `sigma/band` of reward, so the tolerance *is* the bar:
`band >= sigma / MAX_REWARD_NOISE`, i.e. `band_sigma >= 1/0.25 = 4.0`.

Also split the single test into two quantities that only shared the word "noise":

- **precision** -- band wide relative to one submission's seed spread; uses sigma.
- **existence** -- band distinguishable from zero at all; uses the standard error of
  the anchors (`sigma/sqrt(n_seeds)`), Bonferroni-corrected for screening k eval
  sets and shipping the best.

Gate A stays: the reference must beat a randomly-initialised control by more than
one sigma, significantly.

This is a guard-band construction, and there is a standard for it. ILAC-G8:09/2019
defines a *decision rule* as one "that describes how measurement uncertainty is
accounted for when stating conformity", and ISO/IEC 17025:2017 cl. 7.8.6.1 requires
a lab to "document the decision rule employed, taking into account the level of risk
(such as false accept and false reject...)". Setting the acceptance limit equal to
the tolerance limit -- what 0005 effectively did -- is what §4.1 calls *simple
acceptance*, also known as shared risk, "because the probability to be outside the
tolerance limit may be as high as 50%".

## Consequences

- Nobody argues about 3 versus 4 again; they argue about how much reward noise is
  acceptable, which is the real question. Change the tolerance and the bar moves
  with it, visibly.
- `pref-reward-model` fails at 3.10 sigma and ships stamped `provisional`.
- An effect can be real and still useless as a reward: `helpful_rs` is significant
  at z = 5.53 and still fails, which is exactly the distinction the two tests exist
  to draw.
- The clip in 0001 hides a mis-set anchor, so `recovery_raw` is recorded uncapped
  beside the capped value. This is what surfaced mol's soft reference: an agent's
  first attempt scored raw recovery 1.280 and 1.472 while both clipped to 1.0.
- The bar is applied to tasks that shipped *before* it existed. A bar that only ever
  rejects the newest thing is not a bar.

## Confirmation

`python common/shipping.py` -- and since this session it exits non-zero on an
unacknowledged failure, which it did not for its first two days.
