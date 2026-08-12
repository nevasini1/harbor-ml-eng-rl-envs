---
id: 0005
title: Shipping bar set at band_sigma >= 3.0 with an absolute band floor
status: superseded
date: 2026-08-09
supersedes: []
superseded_by: 0006
affects: [mol-property-adapt, pref-reward-model]
commits: []
rule_version: 1
---

## Context

An eval set needs some test of whether its band is wide enough relative to seed
noise to carry a reward at all.

## Decision

Two tests: `band_sigma >= 3.0`, and an absolute floor `band >= 0.02`.

## Why this was retired

Both halves were wrong, in different ways, and this record exists so that is
visible rather than buried:

- **3.0 was calibrated against a previous decision, not derived.** It was picked by
  looking at what the repo had already shipped -- bbbp at 4.09 sigma under the
  then-current convention -- and going one notch below. That is not a threshold with
  a justification; it is a threshold with a precedent.
- **The `band >= 0.02` floor was removed after it excluded an eval set the author
  wanted to keep.** The stated reason was sound (bbbp ships on a band of 0.0143, so
  an absolute floor would exclude an eval set already considered good) but the
  *sequence* was motivated reasoning. In clinical trials this has a name --
  outcome switching -- and CONSORT 2010 items 3b and 6b require changes to methods
  and outcomes to be reported "with reasons" precisely because the sequence matters.

`pref-reward-model` passed under this rule at 3.10 sigma and does not pass under
0006. That pass should never have been treated as one.

## Consequences

Retained in `common/shipping.py::SUPERSEDED` as `rule_version: 1`, with the reason,
so an anchor screened under it is machine-distinguishable from one screened under
0006. It is not deleted: PCI RR's rule for a flawed registered analysis is that it
is "still mentioned in the Methods but omitted with justification from the Results".
