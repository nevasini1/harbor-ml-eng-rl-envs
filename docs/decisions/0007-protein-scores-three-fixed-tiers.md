---
id: 0007
title: The protein task scores three fixed tiers instead of normalized recovery
status: accepted (repairable)
date: 2026-08-09
supersedes: []
superseded_by: null
affects: [sciml-protein-regression]
commits: [3dc7c36]
---

## Context

This task was built before the anchor machinery settled, and its Spearman metric was
thought too noisy to normalize against.

## Decision

Score three fixed tiers on Spearman: 0 below `t_weak`, 0.5 up to `t_strong`, 1.0
above. It is the only task in the repo not scored by 0001.

## Why this is marked repairable rather than accepted outright

The tiers do not discriminate. Both were set below the on-contract frozen ceiling of
0.5332, so a submission that never touches the encoder scores 1.0:

| submission | Spearman | tiers | recovery would give |
|---|---|---|---|
| GPU oracle | 0.5733 | 1.0 | 1.00 |
| frozen probe artifact | 0.5358 | 1.0 | **0.09** |
| naive top-2 fine-tune | 0.5169 | 1.0 | **0.00** |

`t_weak` also has three committed values -- 0.3887 in `tiers.json`, 0.4087 in
`calibrate_tiers.log`, and `round(rho, 4)` in the script, which cannot produce
0.4087 -- and was calibrated on 1,500 rows when the agent's train split has 17,922.
`probe_ceiling.json` measures the same probe at full data as 0.4586, above
`t_strong = 0.45`.

`t_strong = 0.45` is a bare literal, which is the one thing
`finalize_anchors.py`'s docstring singles out as not derived by a stated rule.

## Consequences

- The task discriminates; the reward does not. Its own README says so.
- Being off 0001 means it is also off the criterion in 0006: `shipping.py` prints
  `NOT EXAMINED` for it, and `check_reward.py` covers only its measurement record.
- The repair is unblocked and needs no new measurement: `scripts/lpft.json` holds
  base 0.5332, reference 0.5627, ten per-seed rows, and 4.84 sigma, which clears the
  4.0 bar. Deliberately not done yet: the dataset is set aside.

## Confirmation

`common/check_reward.py::sigma-convention` covers `lpft.json` via `MEASURED_BANDS`;
nothing yet checks `tiers.json` against a measured ladder, because there is no rule
relating them. That is the gap this record exists to hold open.
