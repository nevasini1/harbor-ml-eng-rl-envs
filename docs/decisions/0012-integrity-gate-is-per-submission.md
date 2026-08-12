---
id: 0012
title: The integrity gate is per submission, not per eval set
status: accepted
date: 2026-08-12
supersedes: []
superseded_by: null
affects: [mol-property-adapt]
commits: []
---

## Context

README.md states `reward = integrity_gate x mean(recovery)`, which reads as a single
0-or-1 multiplier over the whole reward. That described two of the three
anchor-scored tasks. `qa-sft-adapt` and `pref-reward-model` cache one model
(`global _MODEL`), so any integrity failure raises for every eval set and the reward
really is 0.

`mol-property-adapt` accepts a **separate model per eval set**
(`/app/final_model/<eval_set>/`) and ran the three anti-substitution layers as each
eval set was scored. So substituting a public checkpoint for tox21 alone got tox21
rejected at 0 and bbbp scored normally -- `mean(0.0, 1.0)` = **0.5**, not 0.

Partial credit for a partially substituted submission is not defensible.

## Decision

Split `integrity_checks()` out of `score_eval_set()` and run it over every eval set's
submitted model **before** scoring any of them. One failure rejects the whole
submission, and the reason names the eval set that failed so the zero stays
attributable.

## Consequences

- The half-substituted submission above now scores 0.0, and both eval sets record
  `status: "rejected"` with a reason naming tox21.
- `integrity_checks` is also callable directly, which the regrade helpers rely on;
  `score_eval_set` recomputes it when not handed a pre-computed block.
- README's formula is now true for all three anchor-scored tasks, and says explicitly
  that the gate is per submission and that a 0 may also mean `error` rather than
  `rejected`.

## Confirmation

Demonstrated against the real `grade_eval_sets`: rejecting tox21 alone gave 0.500
before and 0.000 after, with both eval sets carrying the reason.
