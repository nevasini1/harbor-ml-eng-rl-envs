---
id: 0014
title: reward.json carries the aggregate plus one score per eval set
status: accepted
date: 2026-08-12
supersedes: [0008]
superseded_by: null
affects: [mol-property-adapt, qa-sft-adapt, pref-reward-model]
commits: []
---

## Context

The assignment's §3 asks the verifier to write `reward.json` **multi-metric**:
"encodes your aggregate scalar as the primary reward and per-eval-set scores as
additional metrics."

[0008](0008-reward-json-single-key.md) went the other way, on the stated grounds
that "Harbor's default dataset metric raises on a multi-key reward dict". The
vendored Harbor contradicts that. `verifier/verifier.py::_parse_reward_json` is a
bare `json.loads` with no key-count check, and `metrics/base.py::aggregate_reward_dicts`
has an explicit branch for it:

```python
if len(reward_keys) <= 1:
    return {metric_name: aggregate(values)}
return {key: aggregate([... reward.get(key, 0) ...]) for key in reward_keys}
```

Run against Harbor's own function, single-key collapses to `{"mean": 0.734115}` --
the key name is discarded -- while multi-key returns
`{"reward": 0.734115, "arc_easy": 0.866591, "sciq": 0.908397, "openbookqa": 0.427357}`
and means each key across trials. Every recorded job in `jobs/` shows the collapsed
form, so the per-eval-set detail never reached Harbor at all.

What most likely broke originally was not multi-key but the *contents*: the
pre-0008 payload was `{reward, spearman, n_test: 3427, cosine_min, n_tensors_compared: 108}`,
and `n_test` is a count, not a score. Aggregating 3427 as a metric is meaningless
even where it does not error.

## Decision

`reward.json` carries the aggregate under `reward`, plus one key per eval set whose
value is that eval set's **recovery** -- a score on the reward's own [0,1] scale, so
Harbor's mean across trials is meaningful.

Scores only. Raw metrics, cosines, tensor counts, shingle overlap, status and reason
stay in `metrics.json`. An eval set named `reward` raises rather than silently
overwriting the aggregate.

Carried forward from 0008 and still in force: **the reward is written on every
path.** `grade_eval_sets` floors a failing eval set to 0 with a recorded status and
reason, traps grader-level failures and writes 0, and `test.sh` backstops the case
where the process never reached Python. A missing reward file is a trial error, not a
zero.

## Consequences

- `qa-sft-adapt`'s most interesting property becomes visible to anything reading
  Harbor: 0.867 / 0.908 / 0.427, with the shortfall concentrated on the hardest eval
  set. Under the collapsed form a reader saw only `0.734`.
- The aggregate is unchanged, and that is checked rather than asserted: recomputing
  the three recorded multi-eval-set trials through the modified driver reproduces
  0.734115, 0.909654 and 0.864497 exactly.
- `sciml-protein-regression` is unaffected. It writes its own reward and has one eval
  set, so the aggregate already encodes everything a per-eval-set key would.
- Historical `reward.json` files under `jobs/` stay single-key. They are records of
  what happened, not configuration.

## Confirmation

Verified directly against `.research/harbor/src/harbor/metrics/base.py`, both for
k=1 and k=2, before changing anything.
