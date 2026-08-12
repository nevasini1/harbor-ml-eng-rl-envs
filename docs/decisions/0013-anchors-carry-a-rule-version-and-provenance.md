---
id: 0013
title: Anchors carry the rule version that screened them and the commit that wrote them
status: accepted
date: 2026-08-12
supersedes: []
superseded_by: null
affects: [mol-property-adapt, qa-sft-adapt, pref-reward-model]
commits: [aca3293, f1eecbd, c679c9c]
rule_version: 2
---

## Context

The bar moved from 0005 to 0006 and `band_sigma` changed meaning in 0010, but nothing
in a shipped `anchors.json` distinguished an anchor screened under the old rule from
one screened under the new one. The only way to tell was to date the commit that wrote
the file. `criterion_record()` already claimed to be "stamped into every results file
so the rule cannot move silently" -- it recorded the rule's *parameters*, not its
version.

Ten benchmark and eval projects were surveyed for prior art. Not one carries a
machine-readable threshold record: no `measured_at`, `git_commit`, `n_seeds` or
`supersedes` anywhere. RE-Bench keeps its anchors in prose, under three different
names, in two unsynchronised copies.

## Decision

Two sidecar blocks in the same file as the anchors they describe:

- `_criterion` -- `rule_id`, `rule_version`, the sigma definition, and `supersedes`.
  `SUPERSEDED` retains retired rules with the reason each went.
- `_provenance` -- `assembled_at`, the producing `script`, and a git block
  `{commit, dirty, untracked_files, branch}`, after Inspect AI's `EvalRevision`.

`dirty` is the field most implementations omit and then regret: a commit hash alone
is a lie if the tree had uncommitted changes. It means *tracked* modifications --
untracked files are counted separately, because this repo normally has stray run
output under `jobs/` and folding that in made every stamp dirty, which makes the
field useless.

Bump `rule_version` when a test is added, removed or redefined. Changing
`MAX_REWARD_NOISE` is deliberately not a bump: the bar following the tolerance is the
point of 0006, and the tolerance is already recorded.

## Consequences

- `verifier_core.load_anchors` separates `_`-prefixed keys from eval sets. An earlier
  comment argued a sidecar block would fail the verifier closed -- true of the loader
  as written, so the loader is what changed. A file with only sidecar keys still fails
  closed and names the keys it found.
- It deliberately does not invent a measurement date. `assembled_at` is knowable; when
  the seed runs happened is not recoverable for records that already exist, and
  guessing would be worse than omitting.
- qa and pref are stamped `backfilled: true`: their assembler downloads pinned
  checkpoints and could not be run, so the recorded commit is the one that added
  provenance, not the one that assembled them. The file says so.

## Confirmation

`common/check_reward.py::provenance-recorded` requires both blocks and requires the
rule version to be the **current** one -- a stale version means the file needs
re-deriving. Negative-tested: rolling `rule_version` back to 1 fails, and deleting
`_provenance` fails.
