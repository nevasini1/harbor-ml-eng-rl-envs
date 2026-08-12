---
id: 0002
title: base is the ceiling of the no-adaptation class, not a nominated arm
status: accepted
date: 2026-08-09
supersedes: []
superseded_by: null
affects: [mol-property-adapt, qa-sft-adapt, pref-reward-model]
commits: [e6633a3, 7375e5b]
---

## Context

`base` was initially the obvious no-adaptation arm for each track -- a frozen
backbone with a trained head. Measuring the whole ladder showed the arms swap
places: on bbbp a logistic probe on the CLS embedding scores 0.8978 against the
trained head's 0.8934, and on the preference track the ceiling turned out to have no
parameters at all -- "pick the longer response" scored 0.6031 against a full
fine-tune's 0.6042.

Pinning `base` to a nominated arm therefore pays every lazy submission a slice of
the reward for free, because a cheaper method beats the arm the reward calls zero.

## Decision

`base` is the **maximum** over every arm that does not adapt the model, including
arms that are not models. The arm that wins is recorded in `base_arm` rather than
assumed, and the definition string names the whole screened set.

## Consequences

- tox21's base is the trained head; bbbp's is the logistic probe; `arc_easy`'s is
  `head_only` and `openbookqa`'s is `zero_shot`. The rule, not a preference, picks
  each one.
- A max over noisy means is biased upward. On `arc_easy` the two arms are 0.0013
  apart with sigma 0.0154, so which wins is near a coin flip. The bias is in the
  safe direction (a harder zero), and RE-Bench made the same choice deliberately,
  using "the upper end" of a starting score that varied 17-52%. A proper treatment
  would use an upper confidence bound; open question 4 records this.
- Every new eval set needs its trivial class enumerated and measured, not guessed.

## Confirmation

`research/posttrain/finalize_anchors.py` selects `ceiling_arm` by max; the mol
assembler recovers `base_arm` by matching the shipped `base_auc` against the
measured arms rather than naming it.
