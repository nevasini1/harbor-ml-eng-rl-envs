# Post-training tracks — screening and anchor measurement

Two new environments, built on the machinery the mol task already had: the shared
verifier core (`common/verifier_core.py`), the shared regrade driver
(`common/regrade.sh`), and the same reward shape — continuous recovery between two
*measured* anchors, with integrity as a separate multiplicative gate.

| | `tasks/qa-sft-adapt` | `tasks/pref-reward-model` |
|---|---|---|
| post-training stage | supervised fine-tuning | reward modelling (RLHF stage 2) |
| base model | `HuggingFaceTB/SmolLM2-135M` | `distilroberta-base` |
| data | ARC-Easy / SciQ / OpenBookQA | `Anthropic/hh-rlhf` preference pairs |
| metric | log-likelihood ranking accuracy | pairwise accuracy (chance = 0.5) |
| submission | one causal LM | one encoder, `num_labels=1` |

Both are scored with **no generation and no judge model**. A forward pass over held-out
data is a deterministic function of the submitted weights; anything else puts a second
model's noise inside the reward signal.

Measurement runs one container per (arm, seed) on an A10G
(`modal_measure.py`), and the anchors are derived from the recorded ladder by a separate,
GPU-free step (`finalize_anchors.py`) so the rules can be re-read and re-run.

---

## Gate A — do the pretrained weights do any work?

Every arm is repeated with an identically-configured but **randomly initialized** model.
This is the check that killed the Hydro track and shelved the protein one, and it is the
one that is easiest to pass by accident: an eval set can show a clean band between "no
adaptation" and "real adaptation" while the pretrained weights contribute nothing,
because both arms are learning the same surface feature.

### `qa-sft-adapt` — passes on all three eval sets

5 seeds per arm, 600 held-out items per eval set.

| arm | arc_easy | sciq | openbookqa |
|---|---|---|---|
| `zero_shot` | 0.6017 | 0.6483 | 0.3150 |
| `head_only` | 0.6030 ± 0.0154 | 0.6960 ± 0.0067 | 0.3123 ± 0.0103 |
| `sft_full` | 0.6957 ± 0.0071 | 0.8270 ± 0.0082 | 0.3657 ± 0.0090 |
| `random_init` | 0.2437 ± 0.0228 | 0.2390 ± 0.0038 | 0.2197 ± 0.0135 |

| eval set | base | reference | band | band σ | pretraining gain | ships |
|---|---|---|---|---|---|---|
| `arc_easy` | 0.6030 (`head_only`) | 0.6957 | 0.0927 | **6.02σ** | +0.4520 | yes |
| `sciq` | 0.6960 (`head_only`) | 0.8270 | 0.1310 | **15.98σ** | +0.5880 | yes |
| `openbookqa` | 0.3150 (`zero_shot`) | 0.3657 | 0.0507 | **5.63σ** | +0.1460 | yes |

`random_init` lands at chance (0.25 for four choices) on all three, so the band is
bought by the pretrained weights and not by the training recipe.

**The ceiling arm differs by eval set**, which is the whole reason `base` is defined as a
ceiling rather than as a named method: `head_only` — training only the tied output
embedding, never the transformer body — beats the untouched checkpoint by **+0.048** on
sciq, and *loses* to it on openbookqa. Pinning `base` to `zero_shot` would have paid
every head-only submission ~37% of sciq's reward for doing no adaptation at all.

### `pref-reward-model` — failed Gate A on the first cut, and why

The first cut used a natural sample of hh-rlhf. Measured on it (5 seeds, 1,000 held-out
pairs per eval set):

| arm | helpful_base | helpful_rs | online | harmless |
|---|---|---|---|---|
| `frozen_probe` | 0.5640 | 0.5960 | 0.4990 | 0.5630 |
| `frozen_head` | 0.5646 ± 0.0161 | 0.5932 ± 0.0087 | 0.5118 ± 0.0054 | 0.5888 ± 0.0127 |
| `finetune` | 0.6042 ± 0.0139 | 0.6130 ± 0.0102 | 0.5184 ± 0.0160 | 0.5818 ± 0.0237 |
| `random_init` | **0.5938 ± 0.0085** | **0.5955 ± 0.0111** | 0.4763 ± 0.0078 | 0.4290 ± 0.0171 |

(recorded in [`results/rm_ladder_unbalanced.json`](results/rm_ladder_unbalanced.json))

Two things are wrong here, and the second explains the first:

1. **`random_init` is within seed noise of `finetune`.** +0.010 on helpful_base against
   a σ of 0.014. A randomly-initialized encoder does what a pretrained one does, so the
   task measures the training loop and not the model.

2. **A heuristic with no parameters beats every model arm.** "Pick the longer response"
   scores:

   | eval set | longer-wins | mean chars, chosen | mean chars, rejected |
   |---|---|---|---|
   | `helpful_base` | **0.6031** | 254 | 189 |
   | `helpful_rs` | 0.5704 | 363 | 314 |
   | `online` | 0.4450 | 662 | 684 |
   | `harmless` | 0.4116 | 158 | 204 |

   0.6031 against the fine-tune's 0.6042 and the frozen-encoder ceiling's 0.5646. The
   entire measured band was length. It also explains the strangest number in the table:
   every model arm scored *below chance* on `harmless`, because they learned "longer is
   better" from a training file where that held (0.5356) and it is backwards there
   (0.4116).

This is exactly the failure the mol track's second lesson names — `base` must be the
ceiling of the **trivial class**, and the trivial class includes heuristics that are not
models at all. Two fixes, both now permanent:

- `length_only` is a rung of the ladder (`rm_ladder.arm_length_only`), so the heuristic
  can never silently become the ceiling again.
- the split is **length-balanced**: equal numbers of longer-chosen and shorter-chosen
  pairs in the training file and in every holdout, ties dropped. `length_only` now reads
  exactly 0.5000 on all four sets by construction.

Balancing is applied to the training file too, not just the holdouts. Leaving the
training file biased would teach every submission a feature worth nothing at grade time,
which measures whether the agent spotted the trap rather than whether it can post-train.

### `pref-reward-model` — after the fix

Length-balanced split, 38,420 training pairs, ~3,960 held-out pairs per eval set, 5 seeds:

| arm | helpful_base | helpful_rs | online | harmless |
|---|---|---|---|---|
| `length_only` | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `frozen_probe` | 0.6055 | 0.5976 | 0.5496 | 0.5696 |
| `frozen_head` | 0.6082 ± 0.0056 | 0.6079 ± 0.0046 | 0.5610 ± 0.0042 | 0.6219 ± 0.0101 |
| `finetune` — reference | 0.6315 ± 0.0087 | 0.6268 ± 0.0061 | 0.5633 ± 0.0057 | 0.6390 ± 0.0064 |
| `random_init` | 0.5496 ± 0.0061 | 0.5525 ± 0.0102 | 0.5174 ± 0.0040 | 0.4649 ± 0.0102 |

| eval set | base | reference | band | band σ | pretraining gain | ships |
|---|---|---|---|---|---|---|
| `helpful_base` | 0.6082 (`frozen_head`) | 0.6315 | 0.0233 | 2.68σ | +0.0819 | no |
| `helpful_rs` | 0.6079 (`frozen_head`) | 0.6268 | 0.0189 | **3.10σ** | +0.0743 | **yes** |
| `online` | 0.5610 (`frozen_head`) | 0.5633 | 0.0023 | 0.40σ | +0.0459 | no |
| `harmless` | 0.6219 (`frozen_head`) | 0.6390 | 0.0171 | 1.69σ | +0.1741 | no |

**Gate A is fixed.** `length_only` reads exactly 0.5000 everywhere, and the pretraining
gain went from +0.010 — inside the seed noise — to +0.07…+0.17. The pretrained weights
now do the work.

**One eval set of four clears the shipping bar, and only just.** The obstacle is not the
fine-tune, it is how good the *frozen* encoder is: a trained MLP head on mean-pooled
frozen embeddings reaches 0.6082 on `helpful_base` against a full fine-tune's 0.6315.
That is the same shape of finding as the protein track — frozen features are strong, and
adapting the encoder buys less than intuition suggests.

Three things were tried against the thin band. Two helped and one did not:

| change | result |
|---|---|
| 8,000 → 38,420 training pairs | helped: `helpful_base` band 0.0187 → 0.0233, and the frozen probe rose too (0.5778 → 0.6055) |
| 1,000 → ~3,960 held-out pairs | helped: it is what took `helpful_rs` from noise to 3.10σ |
| 2 → 3 reference epochs ([`rm_reference_3epoch.json`](results/rm_reference_3epoch.json)) | **did not help**: `helpful_base` 0.6315 ± 0.0087 → 0.6255 ± 0.0104, i.e. lower mean and more noise. The 2-epoch reference was kept. |

The task ships with `helpful_rs` alone. It is honest but it is not strong, and the
weakness is recorded rather than hidden:

- 3.10σ against the 3.0σ bar is a hair. One third of the reward band is seed noise.
  (For scale, the mol task's `bbbp` ships at 4.09σ on a band of 0.0143.)
- It is the best of **four** screened eval sets. Picking the maximum of four marginal
  measurements inflates it; the honest reading is "around 3σ", not "3.10σ".

Both are listed as open questions in the top-level README.

---

## What the reward's noise floor is made of

On the first RM cut, the seed-to-seed spread of every arm was ~0.015 at 1,000 held-out
pairs. Binomial noise at n=1,000 and p≈0.58 is 0.0156. The spread was almost entirely
**measurement**, not training instability — so the holdout was re-cut four times larger
(~3,960 pairs). That is what took `helpful_rs` from noise to 3.10σ; it is the single
change that made this task shippable at all.

That trade is bounded by verifier runtime, and the bound is tighter than a micro-benchmark
suggests. Two numbers, both measured at the 4-thread setting the images pin:

| | micro-benchmark, first 128 rows | **real run, full image, end to end** |
|---|---|---|
| `pref-reward-model` | 36 forwards/s → ~3.7 min/eval set predicted | **1,933 s for two eval sets** (≈16 min each, 3,990 + 3,980 pairs) |

The micro-benchmark was wrong by a factor of four. It sampled the first 128 rows, which
are shorter than the corpus average, so almost every batch padded to well under the
256-token cap; the full holdout pads to the cap far more often. The end-to-end figure
also includes what the micro-benchmark left out — model load, four integrity layers, and
a contamination scan that fingerprinted 199,077 windows.

That number is the constraint on the design: `test.sh` allows 3,000 s, so this task could
ship at most **two** eval sets at this holdout size, and it ships one. Measure the
verifier the way it actually runs before choosing a holdout size.

**The size of the held-out set is a reward-design parameter, not a detail** — it sets the
noise floor of every score the environment will ever produce, and what bounds it is
verifier runtime rather than anything about the task.

---

## What the screening cost, and why that shaped the design

All *feasibility* measurement is CPU-only on 8 cores, which is what the tasks give the
agent. Measured there:

| model | training throughput | note |
|---|---|---|
| `distilroberta-base` (82M) | ~2.9 pairs/s at 256 tokens | 1,350 pairs = 474 s/epoch |
| `google/electra-small-discriminator` (13.5M) | ~2.4× faster per token | screened, not used |
| `SmolLM2-135M` | ~2.1 seq/s at 256 tokens | |

Apple MPS was tried and abandoned: a `distilroberta-base` fine-tune that takes 8 minutes
per epoch on 8 CPU threads made no measurable progress in 20 minutes on MPS, and a
SmolLM2-135M step benchmark did not complete in 2 minutes.

This is why the reference recipes are sized the way they are. A task whose reference
needs three hours of the agent's four-hour budget leaves no room to experiment, and an
environment that cannot be iterated in is not measuring engineering.

| task | reference recipe | measured CPU cost on 8 cores |
|---|---|---|
| `qa-sft-adapt` | 3,986 items × 3 epochs, bs 16 | **945 s end to end in the real Harbor agent container** (predicted 17 min from 1.53 s/step locally). Leaves ~3.7 h of the 4-hour budget to experiment. |
| `pref-reward-model` | 38,420 pairs × 2 epochs, bs 8 | 69,156 pair-steps at the measured 2.85 pairs/s = **6.7 h — infeasible**, which is why this task is specified with a GPU |

The preference task's CPU infeasibility is not a compute complaint, it is a *consequence
of the Gate-A fix*. Length-balancing removed the shortcut, and a model that has to judge
content instead of length needs far more than the 8,000 pairs that fitted in a CPU
budget: the frozen probe alone moves from 0.5778 to 0.6055 on `helpful_base` between
8,000 and 38,420 training pairs. Keeping the task CPU-only would have meant keeping it at
a data scale where nothing separates the arms.

---

## Reproducing this

```bash
V=research/.venv/bin/python
$V research/posttrain/fetch_data.py            # corpora, recorded with sha256 pins
$V research/posttrain/pin_models.py            # checkpoint revisions + file hashes

# The locked splits, exactly as shipped. Four eval sets are cut for the preference
# track and one ships; which one is a measurement outcome, not a choice made here.
$V research/posttrain/make_splits.py --track qa --qa-train 4000 --qa-test 600
$V research/posttrain/make_splits.py --track rm --rm-train 40000 --rm-test 4000 \
    --rm-eval-sets helpful_base,helpful_rs,online,harmless

modal run research/posttrain/modal_measure.py --track qa --seeds 5   # ~16 containers
modal run research/posttrain/modal_measure.py --track rm --seeds 5   # ~21 containers
$V research/posttrain/finalize_anchors.py --markdown                 # anchors + ship/no-ship
$V research/posttrain/assemble_tasks.py                              # populate the task trees
$V research/posttrain/verify_graders.py                              # verifier regression suite
```

### What only a real agent caught

The fixture suite below passes, and it still missed a false reject that zeroed an honest
submission. `codex` fine-tuned the provided base on `pref-reward-model` and was graded 0.0
on `min per-tensor cosine 0.8541` — a 768-element attention key bias, while all 51 weight
matrices sat at cosine >= 0.9999.

The reason the suite could not see it: its fixtures are the two *extremes* — an untouched
base at cosine 1.0 and a shuffled embedding at 0.007. Nothing in it was an honest fine-tune
trained harder than the oracle, because I wrote both the oracle and the fixtures. The floor
now applies to weight matrices only; 1-D cosines are reported, not gated.

Agent trial results, one trial each, `codex` (gpt-5.6-sol) on Modal:

| task | reward | runtime |
|---|---|---|
| `qa-sft-adapt` | 0.734115 | 2h 05m |
| `pref-reward-model` | 0.0 as graded, **0.864497** regraded | 49m 31s |

### Verifier regression suites

`verify_graders.py` builds each real image and runs it under `--network none` against
fixtures it constructs. The **accept path is asserted first**: a false reject is
indistinguishable, in the reward, from an agent that did nothing.

| fixture | `qa-sft-adapt` | `pref-reward-model` |
|---|---|---|
| `base_unchanged` — provided base, fresh head | **accepted**, reward 0.0 (raw recovery −0.014 / −0.364 / 0.0 recorded, not hidden) | **accepted**, reward 0.0 |
| `shuffled` — one embedding tensor permuted | rejected: `min per-tensor cosine 0.0002 < 0.9` | rejected: `min per-tensor cosine 0.0071 < 0.9` |
| `nan` — one tensor NaN'd | rejected: `non-finite weights in embed_tokens.weight` | rejected: `non-finite weights in embeddings.word_embeddings.weight` |
| `truncated` — config.json deleted | rejected: `missing config.json` | rejected: `missing config.json` |
| `public_twin` — bit-identical `all-distilroberta-v1` | n/a | rejected by **sha256**, the only layer that sees it |
| `laundered` — `SmolLM2-135M-Instruct` + 1e-4 noise | rejected by **nearest-ancestor** | n/a |
| `contaminated` — agent log quoting held-out rows | rejected: shingle overlap | rejected: shingle overlap |

Every case wrote a `reward.json`. That is the invariant that matters most: a missing reward
file is a trial error, not a zero.

### What only a Harbor run caught

`pref-reward-model` was moved to a GPU after length-balancing made 8,000 pairs too few
(see above). Two things did not move with it, and **nothing in the repo could have told
you**:

1. `environment/Dockerfile` still installed torch from
   `--extra-index-url https://download.pytorch.org/whl/cpu`, copied from the CPU-only
   tasks. `torch.cuda.is_available()` is then False on a machine that has a GPU.
2. `solution/train_reference.py` never called `.to(device)` at all — written when the
   task was CPU-only, and correct then.

Either alone is enough. The reference recipe silently runs on CPU at 2.85 pairs/s, which
is **6.7 hours against a 4-hour timeout**: no exception, no wrong number, just a trial
that dies at the wall an hour after the interesting part. The verifier regression suite
cannot see this — it grades a checkpoint and never runs the agent container. Neither can
a config review, because `gpus = 1` is right and the Dockerfile is right *for a different
task*.

The fix is in both places, and the oracle now prints its device and warns if it is CPU.
The general point: **a task's compute class is not one setting.** Changing it means
re-reading the environment image and the reference script, and the only thing that checks
you did is an end-to-end run.

### The anchors survive the round trip

Running the shipped `solution/solve.sh` through
Harbor on Modal — different hardware from the A10G the anchors were measured on — the
`qa-sft-adapt` oracle reproduced its `reference_acc` on all three eval sets, at uncapped
recovery **1.0104 / 1.0738 / 1.0848**. An anchor that a shipped script cannot reproduce
is the mol task's open question; here it is a checked number.

`make_splits.py` needs `research/posttrain/PRIVATE_SEED`. `assemble_tasks.py` refuses to run
if an anchor is missing, because there is no default to fall back on, and it re-runs
`common/sync.py` so a task cannot ship a grader module that has drifted from `common/`.

Files under `results/`:

| file | what it is |
|---|---|
| `qa_anchors.json`, `rm_anchors.json` | the shipped ladders and the anchors derived from them |
| `rm_ladder_unbalanced.json` | the first preference cut, where length was the whole band |
| `rm_ladder_balanced_8k.json` | length-balanced but only 8,000 training pairs — no eval set shipped |
| `rm_ladder_balanced_38k_2ep.json` | the shipped measurement, kept beside the derivation |
| `rm_reference_3epoch.json`, `rm_randominit_3epoch.json` | the 3-epoch reference that was measured and rejected, with its matching control |
| `public_hashes.json` | pinned revisions and file hashes for both bases and their substitution targets |
