---
id: 0010
title: band_sigma divides by the larger arm, not by the two added in quadrature
status: accepted
date: 2026-08-11
supersedes: []
superseded_by: null
affects: [mol-property-adapt, sciml-protein-regression]
commits: [9e28f8c]
---

## Context

`common/shipping.py` defined sigma as `max(base_std, ref_std)` -- deliberately, so
"a noisy `base` cannot be averaged away by a tight `reference`". The mol anchors
predated that rule and used `band / sqrt(base_std^2 + ref_std^2)`. On bbbp the
quadrature also used the frozen *head's* noise (0.0028) even though the shipped
`base` is the deterministic logistic probe (std 0.0) -- so the noise in the ratio was
not the noise of either arm defining the band.

Nothing shipped or failed differently, because quadrature is the more conservative of
the two and both mol eval sets cleared 4.0 either way. What broke was **comparison**,
and these figures are compared constantly: the protein and preference READMEs both
cited bbbp's 4.09 "for scale" against numbers computed the other way.

## Decision

One definition, computed in one place (`shipping.evaluate`), consumed everywhere.
`research/assemble_task.py` derives the verdict through it at assembly time instead
of carrying a precomputed statistic.

| eval set | was | is |
|---|---|---|
| tox21 | 6.48 sigma | 7.62 sigma |
| bbbp | 4.09 sigma | 6.81 sigma |
| meltome (`lpft.json`) | 3.92 sigma | 4.84 sigma |

## Consequences

- bbbp's "4.09 sigma, the tightest band that ships" was quoted repo-wide and was
  never a real number under either convention. Six files of prose were corrected.
- The protein band **clears** the 4.0 bar at 4.84 rather than narrowly missing it at
  3.92. Its repair plan said the opposite; the blocker there is the tiered reward
  (0007), not separation.
- `report()` had been reconstructing sigma as `band/band_sigma` and printing it beside
  files that state the real number. It now prefers what was recorded.

## Confirmation

`common/check_reward.py::sigma-convention` recomputes from the recorded per-arm noise
and fails if any file disagrees -- including measurement records with no shipped
anchors, which is the only reason the protein track is checked at all.
