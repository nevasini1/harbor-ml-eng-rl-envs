---
id: 0009
title: The lineage cosine floor applies to weight matrices only
status: accepted
date: 2026-08-11
supersedes: []
superseded_by: null
affects: [mol-property-adapt, qa-sft-adapt, pref-reward-model, sciml-protein-regression]
commits: [8d42c88]
---

## Context

The lineage check compares each submitted tensor against the base body by float64
cosine and rejects below 0.9. `pref-reward-model`'s agent read the contract, built a
prompt-level validation split, fine-tuned the provided base -- and scored 0.0:

```
encoder lineage check failed: min per-tensor cosine 0.8541 < 0.9
  on encoder.layer.0.attention.self.key.bias
```

Measured over that submission: 51 weight matrices at cosine >= 0.9999, median 1.0000,
and exactly 1 of 100 tensors under the floor -- a 768-element attention key bias
whose entries are near zero, so a functionally irrelevant update rotates it a long
way. Several transformer implementations omit key biases entirely.

## Decision

The floor applies to tensors with `ndim >= 2`. One-dimensional cosines are reported
as `min_vector_cosine` and no longer reject.

## Consequences

- Two things make this worth a record rather than just a fix. **It was
  intermittent**: the mol agent's 1-D minimum was 0.9998 and sailed through, while
  the reward-model agent trained a fresh scalar head through a ranking loss on a GPU
  and moved that bias far. A verifier that zeroes *some* honest submissions, on a
  vector that does not affect the model's output, corrupts the score distribution
  quietly rather than failing loudly. **And the fixture suite could not have caught
  it**: it tested an untouched base at cosine 1.0 and a shuffled embedding at 0.007,
  never an honest fine-tune trained harder than the reference.
- An oracle validates the plumbing; only an agent validates the task. Three agent
  trials found what fifty measurements and a purpose-built adversarial fixture suite
  had missed.
- A known residual: `worst` initialises to the sentinel 1.0 and the gate is
  `worst < cos_floor`, so "no weight matrices were comparable" passes as "every
  matrix matched". The protein grader calls the check with both internal gates
  disabled and consumes that sentinel.

## Confirmation

Regrading `jobs/mol-oracle-modal` still returns 0.909654 with identical per-eval-set
metrics, and `shuffled`, `nan`, `truncated`, `public_twin` and `contaminated` fixtures
are all still rejected.
