---
id: 0008
title: reward.json carries exactly one key, and is written on every path
status: superseded
date: 2026-08-09
supersedes: []
superseded_by: 0014
affects: [mol-property-adapt, qa-sft-adapt, pref-reward-model, sciml-protein-regression]
commits: [3e6ceb1, 7528241]
---

## Context

Harbor's default dataset metric raises on a multi-key reward dict, so putting a
reason string next to the number loses the score entirely -- the exact failure a
diagnostic field was added to prevent.

Separately, a grader killed outright (OOM, verifier timeout, SIGKILL) writes nothing,
and a submission with no reward is indistinguishable from a submission that failed.

## Decision

`reward.json` holds one key. Per-eval-set detail goes to `metrics.json`, reasons to
`reward_meta.json`. The reward is written on every path: `grade_eval_sets` traps
per-eval-set failures and floors them to 0 with a recorded `status` and `reason`,
traps grader-level failures and writes 0, and `test.sh` backstops the case where the
process never reached Python.

## Consequences

- A zero always carries an attributable reason, distinguishing `rejected` (an
  integrity failure) from `error` (a crash, with a traceback) from a genuinely weak
  model at `ok`.
- `reward = sum(scores)/len(scores) if scores else 0.0` -- a failing eval set stays
  in the mean at 0 rather than being dropped, so failure cannot raise the average.
- The protein task lacked the `rm -f reward.json` and the `timeout` the other three
  carried, and shipped a `scripts/test.sh.hardened` that was byte-identical to the
  unhardened live file. Both fixed; that file deleted.

## Confirmation

`common/verifier_core.py::grade_eval_sets` and each task's `tests/test.sh`.

## Superseded

The single-key half of this decision was wrong, and [0014](0014-reward-json-carries-per-eval-set-scores.md)
replaces it. The always-write half still holds and is restated there, which is why
this record is superseded whole rather than amended: a reader should land on one
current record, not two half-current ones.
