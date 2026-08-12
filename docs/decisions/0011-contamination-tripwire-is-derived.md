---
id: 0011
title: The contamination tripwire is derived by rule, not hand-picked
status: accepted
date: 2026-08-12
supersedes: []
superseded_by: null
affects: [mol-property-adapt, qa-sft-adapt, pref-reward-model]
commits: [43efaba]
---

## Context

`t_implausible` flags a score no honest run has produced, for a human to review. The
post-training tracks derived it: `min(0.98, max(best_observed + 0.15, 0.85))`. mol
hand-wrote 0.85 and 0.97, and the protein task 0.75. The mol grader carried a comment
justifying 0.85 by citing the anchors and the best of 25 seeded runs -- a
justification that goes stale the moment either is re-measured, which both since were.

## Decision

Apply the same rule on the mol track. It gives 0.86 and 0.98 against the typed 0.85
and 0.97 -- so the values were right and only their provenance was missing.

Additionally, no grader passes a hardcoded `DEFAULT_T_IMPLAUSIBLE` any more. Every
anchor carries the key, so all four pass `None` and `load_anchors` fails closed with
"the contamination tripwire would silently not run".

## Consequences

- Both values move upward, i.e. the tripwire loosens slightly. No recorded trial
  changes verdict under either.
- The protein grader read `tiers.get("t_implausible", 0.75)` -- the exact silent
  default the shared core refuses, in the one grader not fully on that core, for a key
  `calibrate_tiers.py` does not emit. `load_tiers` now requires it and asserts it sits
  above `t_strong`, the analogue of the core's `t_imp > reference`.
- `tiers.json`'s stated justification still cites 0.546 +/- 0.005 (n=8) as the frozen
  probe, which 0003 established is the off-contract mean-pooled arm. The margin to
  0.75 is wide either way, but the justification rests on a retracted measurement and
  is annotated as such.

## Confirmation

`research/assemble_task.py` derives it and refuses a value at or below `reference`.
