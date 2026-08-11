# `mol-property-adapt`

Encoder adaptation as an RL environment. The agent gets `DeepChem/ChemBERTa-77M-MLM` — a
3.4M-parameter RoBERTa over SMILES with no task head — labelled molecules for two
property-prediction tasks, and 4 hours on 8 CPUs with no GPU. It must produce the best
predictors it can for both.

**Status: active.** This is the task the rest of the repo is built from: its grader became
[`common/verifier_core.py`](../../common/verifier_core.py), shared by every other task.

## How it is scored

Mean ROC-AUC on a held-out **region of chemical space** — a contiguous neighbourhood of
structurally similar molecules removed whole, so the holdout is a real distribution shift
rather than a random sample.

```
recovery = clip((auc − base) / (reference − base), 0, 1)
reward   = integrity_gate × mean(recovery over the two eval sets)
```

## Anchors

Measured over 5 seeds on the private split
([`research/results/anchors_private.json`](../../research/results/anchors_private.json)):

| eval set | train | test | base | reference | band | separation |
|---|---|---|---|---|---|---|
| `tox21` (12 assays) | 2,000 | 1,566 | 0.6341 | 0.7019 | 0.0678 | **7.62σ** |
| `bbbp` (1 label) | 1,631 | 407 | 0.8978 | 0.9121 | 0.0143 | **6.81σ** |

`base` comes from a **different rung on each set** — a trained head on tox21 (0.6341,
beating a 0.5822 probe), a logistic probe on bbbp (0.8978, beating a 0.8934 head). That is
the point of "ceiling of the trivial class, not one member of it": taking the head on both
would have set bbbp's base 0.0044 low and paid every head-only submission ~31% of the
reward for free.

![effort ladder](../../research/results/anchor_ladder.png)

## Integrity

Three layers on disjoint inputs — architecture-config hash, sha256 against pinned public
checkpoints, per-tensor float64 cosine against the base encoder — plus **InChIKey
contamination**: the verifier computes the InChIKey of every molecule it can parse out of
what the agent submits, including `logs/agent/train_log.txt`, and rejects an eval set if a
held-out molecule appears. Matching on InChIKey rather than the SMILES string is
deliberate; the same molecule has many valid spellings.

The cosine layer deliberately **allows** an unmodified encoder. Freezing the backbone and
training only a head is legitimate; the anchors are what make it score zero.

## Run it

```bash
harbor run -c tasks/mol-property-adapt/configs/job-modal.json --agent oracle
./tasks/mol-property-adapt/scripts/regrade.sh --all
```

The oracle scored **reward 0.909661** through Harbor (`jobs/mol-oracle-modal/`):

| eval set | status | AUC | recovery | raw | overlap | tensors |
|---|---|---|---|---|---|---|
| `tox21` | ok | 0.6896 | 0.8193 | 0.8193 | 0 | 53 |
| `bbbp` | ok | 0.9158 | 1.0000 | **1.2575** | 0 | 53 |

The `raw` column is why the uncapped recovery is recorded at all: on bbbp the oracle beat
the reference by 26%, and the capped `recovery` of 1.0000 hides that completely. Logging
the uncapped value is how a mis-set anchor becomes visible instead of silently flattening
to 1.0.

## An agent run, and what it says about the reference

`codex` (gpt-5.6-sol) scored **reward 1.0** in 1h 22m, reading only `instruction.md`
(`jobs/mol-codex-modal/`):

| eval set | AUC | base | reference | recovery | **raw** |
|---|---|---|---|---|---|
| `tox21` | 0.7209 | 0.6341 | 0.7019 | 1.0 | **1.280** |
| `bbbp` | 0.9188 | 0.8978 | 0.9121 | 1.0 | **1.472** |

It beat the reference on both sets, and its tox21 score exceeds the best of the 25 seeds
(0.7111) that set that anchor. Contamination overlap 0 — the InChIKey scan parsed one
molecule out of its output and correctly did not flag it. It wrote `train_log.txt`, so
rule 4 is followable.

**This is a problem for the task, not a triumph.** Both eval sets clipped at recovery 1.0,
so the reward no longer discriminates at the top: a stronger agent and this one score
identically. The `raw` column is the only reason that is visible at all.

## The reference was re-measured, and could not be raised

The obvious repair is a stronger `reference`. Three candidate recipes were measured against
the private split, 5 seeds each
([`research/modal_mol_reference.py`](../../research/modal_mol_reference.py) →
[`mol_reference_candidates.json`](../../research/results/mol_reference_candidates.json)):

| arm | tox21 | bbbp |
|---|---|---|
| `current` — the shipped recipe, as a control | **0.6896 ± 0.0037** | **0.9151 ± 0.0077** |
| `grouped` — best epoch chosen on held-out scaffold *groups* | 0.6858 ± 0.0050 | 0.9079 ± 0.0048 |
| `stronger` — the agent's LRs, cosine schedule, class weights | 0.6787 ± 0.0217 | 0.9122 ± 0.0044 |

**Neither candidate beats the control on either eval set.** Scaffold-grouped validation —
the change the agent's own log pointed at hardest, and the one defensible as a *defect* fix
rather than a tuning preference — makes it worse on both. The agent's hyperparameters make
tox21 worse and far noisier (±0.0217; two seeds collapse to ~0.65).

So the reference cannot currently be raised on evidence. Editing it upward would produce a
number no script here emits, which is the failure this repo shelved the protein task for.
The reward's top-end blindness is real, but the fix is not a bigger anchor.

**The first run of this was void, and the control arm is the only reason that is known.**
It had batch 16, a 10% validation slice, and a loss normalised over all label slots rather
than observed ones. Under it, `stronger` looked like a clean tox21 win — 0.6990 against
0.6830. The ordering **reversed** once the recipe was transcribed correctly. Without a
control reproducing the shipped number, that run would have shipped the opposite conclusion.

## Known issue: the reference is reproducible, but not from the oracle

`reference_auc = 0.7019` is sound. Re-running
[`modal_legal_anchors.py`](../sciml-protein-regression/scripts/modal_legal_anchors.py)
unmodified returns **0.7024**, with seeds 0/2/3/4 matching the committed values to four
decimals and only seed 1 flipping its best-val epoch
([`legal_anchors_rerun_2026-08-11.json`](../../research/results/legal_anchors_rerun_2026-08-11.json)).

What does not reproduce is the anchor *from the shipped solution*. Four measurements of what
is, on paper, one recipe:

| source | tox21 |
|---|---|
| `modal_legal_anchors.py` — defines the anchor | 0.7019 / 0.7024 |
| `solution/train_reference.py` — shipped oracle, CPU | 0.6897 |
| independent reimplementation, A10G, 5 seeds | 0.6896 ± 0.0037 |
| the oracle's saved checkpoint, re-scored directly | 0.689651 |

The two groups do not overlap across 5 seeds each. That is why the oracle scores 0.9097
rather than 1.0: **the task's upper anchor is defined by a script the shipped solution
cannot reach.**

Five hypotheses have been eliminated by measurement, so they need not be re-guessed:

| hypothesis | how it died |
|---|---|
| loss normalisation (`/M.sum()` vs `/M.numel()`) | the anchor script and the reimplementation now share `/M.sum()` and still differ |
| the training split moved after the anchor was measured | the deterministic frozen-logreg arm re-runs bit-exactly at 0.5822; it could not if the data had changed |
| AUC task-skip rule (`obs < 10` vs 2-unique) | both average over all 12 tox21 assays on this test set |
| eval batch size (128 vs 64) | identical AUC to 6 decimals on the same checkpoint |
| score transform (raw logits vs float32 sigmoid) | bit-identical AUC; logit range [−7.98, 4.15], zero saturated cells |

The non-overlapping seed ranges point at an evaluation-side systematic, but every
evaluation-side candidate is now excluded — so the difference is training-side despite that.
Tracked as open question 1 in the root [README](../../README.md#open-questions).

Split construction and anchor measurement live in [`research/`](../../research/SPIKE_RESULTS.md).
