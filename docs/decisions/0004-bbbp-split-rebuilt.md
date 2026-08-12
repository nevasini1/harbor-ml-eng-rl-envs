---
id: 0004
title: The bbbp private split is rebuilt; a cluster-level split is rejected
status: accepted
date: 2026-08-09
supersedes: []
superseded_by: null
affects: [mol-property-adapt]
commits: [ef5f524, 6ffc7fb]
---

## Context

The shipped bbbp private split measured nothing: it was 89% positive with roughly
44 negatives, so its fine-tune noise was 24.7% of the band against tox21's 8.1%.
A band that wide relative to noise scores the seed, not the submission.

Separately, a cluster-level (scaffold-group) split was measured to see whether a
more principled split restores headroom.

## Decision

Rebuild the bbbp private split; keep the scaffold split rather than moving to a
cluster-level one. The cluster split was measured and **rejected on evidence**: it
did not restore headroom. Both scientific tracks found the more principled split
made the task *easier*, not harder.

## Consequences

- bbbp's held-out set went from 204 to 407 rows; `n_test` in the anchors and the
  shipped CSV agree, and are checked.
- Anchors are functions of the split, so both had to be re-measured. This is the
  rule now stated as lesson 8.
- The random-init control for mol was measured on the *old* splits (bbbp n_test 204,
  tox21 6,258/783) and never re-run. Gate A on this task is therefore unmeasured
  rather than passed -- see 0006's consequence and the open question.
- "Removing leakage" and "increasing difficulty" are separate things, and this is
  the measurement that separated them.

## Confirmation

`research/results/variance_bbbp.json` and `tasks/sciml-protein-regression/scripts/
bbbp_split_v2.json` hold the measurements; `common/shipping.py` prints
`ships (gate A NOT MEASURED)` for both mol eval sets rather than a bare `ships`.
