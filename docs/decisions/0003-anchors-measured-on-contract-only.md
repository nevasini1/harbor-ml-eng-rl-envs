---
id: 0003
title: Anchors are measured only over arms a legal submission could produce
status: accepted
date: 2026-08-09
supersedes: []
superseded_by: null
affects: [mol-property-adapt, sciml-protein-regression]
commits: [031a886]
---

## Context

Both anchors on the mol task were first measured with mean-pooled embeddings. The
submission contract is `RobertaForSequenceClassification` + `.logits`, whose head
reads the CLS token: `x = features[:, 0, :]`. Mean pooling is therefore not
expressible in any submission an agent may hand in.

The consequence is worse than an offset. `base` became a ceiling over methods no
agent could reach, and `reference` became a target no agent could hit -- the reward
was normalized between two unreachable points. The same defect appeared
independently on the protein task, where a mean-pooled frozen probe at 0.546 was
used as the ceiling while the legal CLS frozen head is 0.5332.

## Decision

An arm may define an anchor only if a legal submission could produce it. Anchors
were re-measured inside the contract: tox21 base 0.6341 / reference 0.7019, bbbp
0.8978 / 0.9121. Off-contract measurements are retained under
`superseded_off_contract` with the reason, not deleted.

## Consequences

- The mol anchors moved: base 0.643 -> 0.6341, reference 0.7324 -> 0.7019.
- "Which arms are legal" became a property of the task that has to be written down,
  which is what `contract` in `anchors_private.json` records.
- `research/lock_lowdata_anchors.py` still writes the off-contract anchors and is
  marked STALE -- DO NOT RUN. That warning is now enforced rather than requested:
  the assembler refuses a `base_definition` describing a mean-pooled arm (0012's
  sibling change, commit 1ee17c5).

## Confirmation

`research/assemble_task.py` raises on a mean-pooled `base_definition` and on an
anchor with no `measured_arms_n5_legal` block.
