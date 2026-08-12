---
id: 0001
title: Reward is normalized recovery between two measured anchors
status: accepted
date: 2026-08-08
supersedes: []
superseded_by: null
affects: [mol-property-adapt, qa-sft-adapt, pref-reward-model]
commits: [f5bec43]
---

## Context

The task is "can this agent adapt a pretrained model", not "can it call `fit()`". A
raw metric cannot express that: 0.70 ROC-AUC on tox21 is excellent or trivial
depending entirely on what a submission that does nothing would have scored. Any
reward built directly on the metric measures the dataset's difficulty as much as
the submission.

## Decision

Per eval set, normalize the raw metric onto [0,1] between two **measured** anchors:

```
recovery = clip((metric - base) / (reference - base), 0, 1)
reward   = integrity_gate x mean(recovery over eval sets)
```

`base` earns 0, `reference` earns 1. Both are measurements on the private split
with a committed script behind them, not targets chosen by judgement.

This is the same construction as METR's RE-Bench (§3.2.2), which normalizes between
a *starting score* and a *reference solution score*. Their version has no upper
clip; ours clips at 1.0 and records the uncapped value alongside (see 0006's
consequence about `recovery_raw`).

## Consequences

- A reward is comparable across eval sets with different metrics and difficulties.
- The reward is only as good as the two anchors, which moves all the risk into
  measuring them. Most of `research/` exists because of this decision.
- An anchor is now a shipped artifact with a correctness requirement, which is what
  makes 0003, 0006, 0010, 0011 and 0013 necessary.

## Confirmation

`common/verifier_core.py::recovery`, and `common/check_reward.py` asserts every
shipped anchor still matches the measurement record behind it.
