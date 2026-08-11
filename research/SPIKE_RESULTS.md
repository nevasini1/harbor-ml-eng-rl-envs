# Phase 0 spike results — CPU-only ML engineering env

Two tracks were tested. **The protein track failed both gates and was abandoned.
The molecular property track passed both and is the recommendation.** The protein
findings are kept below because they explain why the pivot was necessary.

---

# PART 2 (current recommendation) — molecular property prediction

Base model: [`DeepChem/ChemBERTa-77M-MLM`](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)
— a RoBERTa over SMILES, **3.43M parameters**, 3 layers, hidden 384 (the "77M" is
pretraining molecules, not parameters). Data: MoleculeNet via the DeepChem S3 mirror,
Bemis-Murcko scaffold splits, 80/10/10. Metric: mean ROC-AUC over tasks.

Dependency risk is gone: this needs only `transformers` + `rdkit`. No PyTorch
Geometric, and no Google-Drive-hosted checkpoints (which is what ruled out the
`snap-stanford/pretrain-gnns` GIN weights).

## Gate A — headroom: PASS

The decisive test is not "does the model beat a classical baseline" but **"do the
pretrained weights do any work"** — the check that would have caught Hydro instantly.
So every fine-tune was repeated with an identically-configured but randomly-initialized
model.

| dataset | morgan+logreg | morgan+RF | ChemBERTa frozen | ChemBERTa fine-tune | random-init fine-tune | pretraining gain |
|---|---|---|---|---|---|---|
| **bbbp** | 0.6722 | 0.7050 | 0.7264 | **0.7436** | 0.6666 | **+0.0770** |
| **tox21** | 0.7106 | 0.7052 | 0.7197 | 0.7061 | 0.6636 | **+0.0425** |
| clintox | 0.8219 | 0.7216 | 0.9867 | 0.9812 | 0.9964 | −0.0152 |
| bace | 0.8342 | **0.8774** | 0.7835 | 0.7654 | — | — |
| sider | 0.6103 | 0.6253 | 0.6112 | 0.5679 | — | — |

**BBBP is the primary eval set.** It gives a clean, strictly increasing ordering —
random-init 0.667 < Morgan+RF 0.705 < frozen probe 0.726 < fine-tune 0.744 — so every
step of genuine work is rewarded, and the pretrained weights are worth +7.7 AUC points.
That matches the +7.2 published for pretrained GNNs.

**Tox21 is the secondary eval set**, and it is interesting precisely because the naive
recipe *loses*: a 12-epoch fine-tune (0.7061) lands below the frozen probe (0.7197),
while still beating random-init by +4.3 points. Its learning curve was rising
monotonically and had not converged at 12 epochs, so the headroom is real but only
reachable with a better training recipe. That is exactly the behaviour we want an agent
to be tested on.

**ClinTox must be dropped**: random-init (0.9964) beat pretrained (0.9812). The task is
near-saturated and learnable from SMILES syntax alone, so it measures nothing.

**BACE and SIDER favour fingerprints** over ChemBERTa. They are useful as robustness or
trap eval sets, not as the primary reward.

## Private-split gap widening (locked)

Full-data region holdouts only gave **+0.023** base→reference headroom. Screening
train sizes on the hard region (anchor 2576) found a low-data sweet spot:

| train N | frozen probe | fine-tune | gap |
|---|---|---|---|
| 500 | 0.601 | 0.643 | +0.042 |
| 1000 | 0.655 | 0.667 | +0.011 |
| **2000** | **0.603** | **0.691** | **+0.088** |
| 4000 | 0.655 | 0.706 | +0.051 |
| full (~6.3k) | 0.678 | 0.701 | +0.023 |

**Locked task shape:** tox21 only, chemical-region holdout test (1,566 mols),
agent gets 2,000 labelled train molecules. Anchors:
`base_auc=0.6027`, `reference_auc=0.6907`, **gap=+0.088**.

> **Superseded.** Both anchors above are off-contract and both eval sets were
> re-measured; see *Anchors, re-measured inside the contract* below.

## Anchors, re-measured inside the contract

The anchors above could not be reached by a legal submission. `tests/grade.py`
loads with `AutoModelForSequenceClassification` and reads `.logits`, and the base
is `model_type: roberta`, so a submission *is* a `RobertaForSequenceClassification`
whose head reads `features[:, 0, :]` — the CLS token. Both anchors had been
measured with **mean-pooling**, which that architecture cannot express. Setting
`reference_auc` from a mean-pooled fine-tune (0.7324) would have put reward 1.0
permanently out of reach.

Two further errors compounded it. `base_auc=0.6027` was a logistic probe, while a
trained head reaches higher at the same zero effort — so head-only collected ~46%
of the reward for freezing the backbone. And `reference_auc=0.6907` was a single
run: five seeds of that same recipe average 0.7006, so ~0.010 of the apparent
shortfall was seed noise rather than recipe.

Every arm below is a legal submission, 5 seeds, on the private split:

| eval set | base | reference | band | separation | measurement |
|---|---|---|---|---|---|
| **tox21** | 0.6341 | 0.7019 | 0.0678 | **7.62σ** | `scripts/legal_anchors.json` |
| **bbbp** | 0.8978 | 0.9121 | 0.0143 | **6.81σ** | `scripts/bbbp_split_v2.json` |

`base` is the **ceiling** of the methods that do not adapt the encoder, not any
single one of them. On tox21 the trained head wins (0.6341 vs a 0.5822 probe); on
bbbp the probe wins (0.8978 vs 0.8934). Taking the head on both would have set
bbbp's base 0.0044 low and paid every head-only submission ~31% for free.

**bbbp is back, on a rebuilt split.** The shipped bbbp split was the rare-scaffold
tail this file already records as rejected, and it measured a band of +0.0006 at
0.34σ — frozen logreg 0.9671, frozen head 0.9668, fine-tune 0.9677, all inside
seed noise. A Tanimoto anchor-region holdout was rebuilt instead, screening 40
candidate anchors and selecting on **band ÷ pooled noise** rather than raw band,
subject to at least 100 minority-class test molecules. That balance floor is not
cosmetic: AUC variance is governed by the scarcer class, and the first attempt
maximised raw band and won with a test set 89% positive (~44 negatives), whose
fine-tune noise was 24.7% of band against tox21's 8.1%.

**Separation is the criterion**, not per-arm noise as a share of band. The two are
not independent, so demanding σ ≥ 3 *and* per-arm ≤ 10% implicitly demands σ ≳ 7.
Both halves of that argument are now retired: the bar is 4.0, derived from
`MAX_REWARD_NOISE = 0.25`, the per-arm test was removed, and σ divides by the larger
of the two arms rather than by their pooled noise — under which tox21 is 7.62σ.

**End to end**, the oracle scores **0.9097** through the real agent image on CPU
(6.9 min) and the real verifier image: tox21 recovery 0.819, bbbp 1.000 (uncapped
1.258). One caveat stands: the tox21 oracle landed at 0.6897, below the minimum of
the five seeds that set the anchor (0.6967–0.7111), because `train_reference.py`
and the anchor arm are the same recipe in two implementations. Either re-measure
the anchor from the shipped script or drop the claim in its docstring that the
anchor is defined as what it produces.

Figure: [`results/anchor_ladder.png`](results/anchor_ladder.png).

## Gate B — budget: PASS decisively

| workload | wall clock (8 threads) |
|---|---|
| embed 2,050 molecules (BBBP) | 1.8–14 s (145–1,160 mol/s) |
| embed 7,831 molecules (tox21) | 18 s (429 mol/s) |
| **full 12-epoch fine-tune + eval, BBBP** | **~2.5 min** (134 mol/s) |
| full 12-epoch fine-tune + eval, tox21 | ~11 min (112 mol/s) |
| Morgan fingerprints + RF baseline | 0.4–43 s |

A complete train-and-evaluate cycle on BBBP is **two and a half minutes**. In a 4-hour
budget an agent can run on the order of 50–90 full experiments, or ~20 on tox21. This is
a genuine optimization loop — the thing neither Hydro nor Meltome could provide.

## Reward variance: PASS

Five training seeds on BBBP, 12 epochs each:

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| final AUC | 0.7411 | 0.7397 | 0.7404 | 0.7374 | 0.7452 |

mean 0.7408, **std 0.0025**, range 0.0078.

Seed noise is ~30x smaller than the 0.074 base-to-reference gap, and ~6x smaller than
even the tighter frozen-probe-to-fine-tune gap of 0.014. Comfortably within the brief's
requirement that measured variance sit well below the anchor gap.

## Open risk

MoleculeNet labels are public, so a private scaffold re-split changes the partition but
not the labels — the same structural exposure every public benchmark has. Planned
mitigations: private split seed never released, test molecules never placed in the
agent's container, verifier on `no-network` with data baked into the image, and an
InChIKey train/test overlap check at grade time. This needs to be documented honestly
rather than overclaimed.

---

# PART 1 (abandoned) — protein fitness regression

All numbers measured on 8 CPU threads (Apple Silicon, 8 cores), `facebook/esm2_t6_8M_UR50D`
@ `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`, FLIP2 Zenodo record 18433203 (CC-BY 4.0).
Metric is Spearman on the split's own test set.

## Verdict

The protein fitness track fails on CPU, for a structural reason rather than bad luck.
Two datasets were tested and they fail in opposite, complementary ways.

## Gate A — Hydro: FAILED (pretrained model is dead weight)

Hydro is a combinatorial library: three wild-type backbones (57/63/65 aa, matching
P06241/P01053/P0A9X9), each with exactly **7 randomized core positions drawn from the
same 5 hydrophobic residues** (F/I/L/M/V). So the learning problem is a 35-parameter
regression with 6,463–9,972 observations, and a protein language model has nothing to add.

`random_split` (train 19,948 / test 4,987):

| method | Spearman |
|---|---|
| one-hot ridge (35 features) | **+0.755** |
| ESM frozen-embedding ridge | +0.760 |
| aa-composition ridge | +0.389 |
| ESM zero-shot (masked marginal) | −0.074 |

`to_P06241` — wild-type transfer, train on the 63/65 aa backbones, test on 57 aa:

| method | Spearman |
|---|---|
| ESM zero-shot | **+0.268** |
| aa-composition ridge | +0.092 |
| ESM frozen-embedding ridge | +0.013 |
| one-hot ridge | undefined (features do not transfer) |
| ESM full fine-tune, 4 epochs | +0.069 → +0.100 → +0.078 → **−0.007** |

Every supervised approach scored *below* the untrained base model. The fine-tune
memorized the two training backbones and did not transfer.

A 27-cell sweep over two further regimes confirmed it: in **low-label transfer**
(k = 25…1000 labelled target-backbone variants) and **mutation-order extrapolation**
(train ≤3 or ≤4 mutations, test the rest), one-hot ridge won or tied in every cell.
At k=25 on the 57 aa backbone, one-hot scored 0.572 against ESM's 0.418.

The intended solution to this task would have been `sklearn` on 35 features. This is
FLIP2's own headline finding — simpler models often match fine-tuned pLMs — reproduced
sharply.

## Gate B — Meltome-mixed: FAILED (too slow to iterate)

Meltome-mixed is the opposite regime and the right one on quality grounds: 23,340
distinct proteins with no shared coordinate system, so position one-hot is not even
definable and signal must come from sequence modelling. Published headroom is wide
(ridge-on-one-hot 0.17 vs ESM-1b 0.68).

It fails on cost. Median sequence length is 413 aa (p95 1,377; max 35,213). Truncated
to 512 tokens, embedding throughput degraded from 20 seq/s on the short tail to
**6 seq/s** on the bulk. 19,232 of 27,951 sequences took 53 minutes, projecting to
**~85 minutes for one forward pass** on 8 cores.

That is ~35% of a 4-hour all-inclusive budget for a single featurization with no
training and no iteration. Backward passes are roughly 3x forward, putting a full
fine-tune at ~4 h/epoch.

## Why this generalizes

Protein fitness benchmarks partition into exactly these two shapes:

- **Short sequences** (fast on CPU) exist because a fitness landscape is measured by
  mutagenizing *one* protein. That makes them combinatorial libraries over a handful of
  positions, where a pretrained sequence model is worthless.
- **Diverse sequences** (where pretraining is load-bearing) mean full-length proteins of
  400+ residues, which are too slow to iterate on with CPU-only compute.

Within FLIP2 this holds across the board: the short datasets (Hydro 57-65 aa, PDZ3
101-109 aa, GB1 265 aa with only 4 varying positions, NucB ~110 aa) are all single-backbone
mutational libraries, while the diverse ones (Meltome, SCL, secondary structure) are all
long.

## Throughput reference (ESM-2-8M, 8 threads)

| workload | throughput |
|---|---|
| embed, 57–65 aa | 150–190 seq/s |
| embed, 512 aa | 6–20 seq/s |
| full fine-tune, 63 aa | 51 seq/s (~5 min per 15k-sequence epoch) |

## What did succeed and carries forward

The verifier is built and passes its full adversarial matrix. Every case produced a
valid reward and none raised out of the grader:

| fixture | status | caught by |
|---|---|---|
| base (untouched encoder + fresh head) | ok | — (legitimate) |
| constant output | ok, reward 0 | degenerate-prediction guard |
| ESM-2-35M substituted | rejected | architecture hash (`hidden_size` 320→480) |
| weights shuffled within tensors | rejected | lineage, min cosine **−0.1249** vs 0.90 floor |
| truncated safetensors | error, reward 0 | `SafetensorError` caught |
| empty dir / missing config | rejected | config presence check |
| 100k-deep nested config.json | rejected | `RecursionError` caught |
| config.json symlinked to /dev/zero | rejected | symlink refused before read |

Design points worth keeping regardless of dataset:

- The lineage check must **allow an unmodified encoder**, because freezing the backbone
  and training only a head is a legitimate strategy. "Weights must have moved" is not a
  valid requirement, so the sha256-vs-public layer does the real anti-substitution work.
- Per-tensor float64 cosine, not a single global cosine. The shuffled fixture confirms
  the −0.12 vs 0.90 margin is enormous.
- HF exposes sha256 only for LFS files; `config.json` returns `None`, so the verifier
  must hash small files itself.
- For this base model the public-hash layer is largely redundant with the architecture
  layer, since no other public checkpoint shares ESM-2-8M's shape. That layer matters far
  more for an LLM task (e.g. Qwen base vs its instruct sibling, which share all seven
  core architecture fields).

## Operational notes

- The sandbox segfaults numpy on import; all compute must run unsandboxed.
- Python block-buffers stdout when not a TTY — run long jobs with `-u` or progress is invisible.
- Dense `RidgeCV` on a 24,817 x 4,000 TF-IDF matrix is a single-threaded SVD that ran
  50+ minutes without finishing; use the sparse CG solver with an explicit validation split.
