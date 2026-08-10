# Harbor ML-engineering RL environments

Agent-evaluation environments for ML engineering: give an agent a pretrained model, a
fixed compute budget and labelled data, and score whether it can *adapt* the model —
not whether it can call `fit()`.

The hard part is not the task. It is the **reward**: making a number that goes up only
when the agent does the work you meant to test, and that says the same thing twice in a
row. Most of this repo is the evidence behind the anchors that number is built from.

---

## The four tasks

| | [`mol-property-adapt`](tasks/mol-property-adapt/) | [`qa-sft-adapt`](tasks/qa-sft-adapt/) | [`pref-reward-model`](tasks/pref-reward-model/) | [`sciml-protein-regression`](sciml-protein-regression/) |
|---|---|---|---|---|
| **status** | **active** | **active** | **active** | **shelved** |
| what it tests | encoder adaptation | supervised fine-tuning | reward modelling (RLHF stage 2) | encoder adaptation |
| base model | ChemBERTa-77M-MLM (3.4M) | SmolLM2-135M | distilroberta-base (82M) | ESM-2-8M |
| data | Tox21 + BBBP, chemical-region holdouts | ARC-Easy / SciQ / OpenBookQA | hh-rlhf preference pairs | FLIP2 meltome-mixed |
| metric | mean ROC-AUC | log-likelihood ranking accuracy | pairwise accuracy | Spearman |
| compute | 8 CPU, 16 GB, 4 h | 8 CPU, 16 GB, 4 h | **1 GPU** + 8 CPU, 4 h | **1 GPU** + 4 CPU, 4 h |
| reward | continuous recovery between two *measured* anchors | same | same | 3 discrete tiers on *fixed* thresholds |
| eval sets | tox21, bbbp | arc_easy, sciq, openbookqa | helpful_rs | meltome-mixed |
| separation | 6.5σ / 4.1σ | 6.0σ / 16.0σ / 5.6σ | 3.1σ — thin, see below | **inverted**: a frozen probe beats a fine-tune |
| oracle through Harbor | 0.9097 | **1.0** | 0.5828 | 1.0 — but so does a frozen probe |

All four tasks have now been run end to end through Harbor with their own shipped oracle —
agent container, artifact hand-off, verifier, reward — and the numbers are in
[What has actually been run](#what-has-actually-been-run). The rewards differ for reasons
worth understanding rather than fixing: 1.0 on `qa-sft-adapt` means the oracle cleared its
reference on every eval set; 0.58 on `pref-reward-model` means it landed 1.3σ under a
reference that is a five-seed *mean*, which a single seed does about half the time; and
1.0 on the protein task means nothing at all, because a submission that never touches the
encoder scores 1.0 there too.

The two post-training tasks are new and share the mol task's verifier machinery rather
than reimplementing it — see [What is shared](#what-is-shared). The protein task is kept
because its negative result is worth reading; its reward is not usable, see
[Why the protein task is shelved](#why-the-protein-task-is-shelved).

---

## The reward

Every active task scores the same way. Per eval set, the raw metric is normalized onto
[0,1] between two measured anchors:

```
recovery = clip((metric − base) / (reference − base), 0, 1)
reward   = integrity_gate × mean(recovery over eval sets)
```

- **`base`** — the score that earns **0**: the *ceiling* of everything that does **not**
  adapt the model. Not any single trivial method — the best of them, including methods
  that are not models at all.
- **`reference`** — the score that earns **1**: a tuned, deliberately ordinary adaptation.
- **`integrity_gate`** — 0 if provenance, contamination or shape checks fail, so a 0
  always carries an attributable reason instead of being indistinguishable from a weak
  model.

Both anchors are **measurements**, taken over 5 seeds on the private split, each
re-derivable from a committed script. The uncapped `recovery_raw` is recorded beside the
capped value, because the clip is what hides a mis-set anchor.

The band between the anchors is the entire scoring range, so its **width relative to
seed noise** decides whether the reward measures the submission or the seed. That ratio
is reported as `band_sigma`, and it is the criterion for whether an eval set ships.

---

## What is shared

The mol task's grader was the only one whose behaviour had been measured end to end
through Harbor, so it became the library rather than being reimplemented three times.

| | what it is |
|---|---|
| [`common/verifier_core.py`](common/verifier_core.py) | the integrity layers, the fail-closed anchor loader, the recovery normalization, and the driver that guarantees a reward is always written |
| [`common/textmatch.py`](common/textmatch.py) | shingle-overlap contamination — the text analogue of the mol task's InChIKey check |
| [`common/regrade.sh`](common/regrade.sh) | re-score a finished trial against a rebuilt verifier image, for every task |
| [`common/sync.py`](common/sync.py) | copies the shared modules into each task's build context; `--check` fails if a copy has drifted |

The extraction is behaviour-preserving, and that is checked rather than asserted:
regrading `jobs/mol-oracle-modal` through the refactored grader returns **0.909654**,
with byte-identical per-eval-set metrics to the original grader run on the same host.
(The recorded 0.909661 came from x86 Modal; the 7×10⁻⁶ difference is arm64 float, and the
*original* grader reproduces it too.)

A Docker build context cannot reach outside itself, so each task's `tests/` holds a
byte-identical copy of the shared modules. `python common/sync.py --check` is what makes
drift loud.

### Four integrity layers, failing on disjoint inputs

1. **architecture-config hash** vs the provided base.
2. **sha256** vs known public checkpoints — repo compared for *equality*, not prefix. A
   prefix test would have waved `SmolLM2-135M-Instruct` through as the base
   `SmolLM2-135M`.
3. **per-tensor float64 cosine** vs the base body. This deliberately *allows* an
   unmodified body: freezing the backbone and training only a head is a legitimate
   strategy, so "weights must have moved" is not a valid requirement. The anchors are
   what make laziness score zero.
4. **nearest-ancestor** (`qa-sft-adapt` only, new): reject a submission closer to a
   same-architecture public sibling than to the base. This is the only layer that catches
   a *laundered instruct checkpoint* — identical config, weights moved by the agent's own
   training, still correlated with the base. Verified: an `SmolLM2-135M-Instruct` copy
   perturbed by 1e-4 is rejected with `mean cosine 1.000000` to the sibling against
   `0.996951` to the base, while the honest base is accepted.

---

## Measured ladders

Every arm below is a legal submission. Anchors come from 5 seeds on the private split;
`base` is always the **best** no-adaptation arm on that eval set, never a nominated one.

### `mol-property-adapt`

| eval set | train | test | base | reference | band | separation |
|---|---|---|---|---|---|---|
| `tox21` (12 assays) | 2,000 | 1,566 | 0.6341 | 0.7019 | 0.0678 | **6.48σ** |
| `bbbp` (1 label) | 1,631 | 407 | 0.8978 | 0.9121 | 0.0143 | **4.09σ** |

`base` comes from a different rung on each set — a trained head on tox21 (0.6341, beating
a 0.5822 probe), a logistic probe on bbbp (0.8978, beating a 0.8934 head). Taking the head
on both would have set bbbp's base 0.0044 low and paid every head-only submission ~31% of
the reward for free.

![effort ladder](spike/results/anchor_ladder.png)

### `qa-sft-adapt`

| arm | arc_easy | sciq | openbookqa |
|---|---|---|---|
| `zero_shot` (submit the base unchanged) | 0.6017 | 0.6483 | 0.3150 |
| `head_only` (output embedding only) | 0.6030 ± 0.0154 | 0.6960 ± 0.0067 | 0.3123 ± 0.0103 |
| `sft_full` — **reference** | 0.6957 ± 0.0071 | 0.8270 ± 0.0082 | 0.3657 ± 0.0090 |
| `random_init` | 0.2437 ± 0.0228 | 0.2390 ± 0.0038 | 0.2197 ± 0.0135 |

| eval set | base | reference | band | separation | pretraining gain |
|---|---|---|---|---|---|
| `arc_easy` | 0.6030 (`head_only`) | 0.6957 | 0.0927 | **6.02σ** | +0.4520 |
| `sciq` | 0.6960 (`head_only`) | 0.8270 | 0.1310 | **15.98σ** | +0.5880 |
| `openbookqa` | 0.3150 (`zero_shot`) | 0.3657 | 0.0507 | **5.63σ** | +0.1460 |

The mol task's seam, in a new place: `head_only` — training only the tied output
embedding, never the transformer body — beats the untouched checkpoint by **+0.048** on
sciq and *loses* to it on openbookqa. Pinning `base` to "the base model scored as-is"
would have paid every head-only submission ~37% of sciq's reward.

### `pref-reward-model`

Length-balanced split, 38,420 training pairs, ~3,960 held-out pairs per eval set:

| arm | helpful_base | helpful_rs | online | harmless |
|---|---|---|---|---|
| `length_only` (pick the longer response) | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `frozen_probe` (Bradley-Terry on frozen embeddings) | 0.6055 | 0.5976 | 0.5496 | 0.5696 |
| `frozen_head` (trained MLP on those embeddings) | 0.6082 ± 0.0056 | 0.6079 ± 0.0046 | 0.5610 ± 0.0042 | 0.6219 ± 0.0101 |
| `finetune` — **reference** | 0.6315 ± 0.0087 | 0.6268 ± 0.0061 | 0.5633 ± 0.0057 | 0.6390 ± 0.0064 |
| `random_init` | 0.5496 ± 0.0061 | 0.5525 ± 0.0102 | 0.5174 ± 0.0040 | 0.4649 ± 0.0102 |

| eval set | base | reference | band | separation | pretraining gain | ships |
|---|---|---|---|---|---|---|
| `helpful_base` | 0.6082 (`frozen_head`) | 0.6315 | 0.0233 | 2.68σ | +0.0819 | no |
| `helpful_rs` | 0.6079 (`frozen_head`) | 0.6268 | 0.0189 | **3.10σ** | +0.0743 | **yes** |
| `online` | 0.5610 (`frozen_head`) | 0.5633 | 0.0023 | 0.40σ | +0.0459 | no |
| `harmless` | 0.6219 (`frozen_head`) | 0.6390 | 0.0171 | 1.69σ | +0.1741 | no |

This task took three tries to become measurable, and the first two failures are the
interesting part.

**It did not measure the model.** The first cut used a natural sample of hh-rlhf. On it, a
heuristic with no parameters — "pick the longer response" — scored **0.6031** on the
helpful holdout, against **0.6042** for a full fine-tune and 0.5646 for the frozen-encoder
ceiling. A randomly initialized encoder reached 0.5938, inside the seed noise of the
pretrained one. The band was length, and nothing else. It also explains the strangest
number in that table: every model arm scored *below* chance on `harmless`, because they
learned "longer is better" from a training file where it held (0.5356) and it is
backwards there (0.4116).

The fix is in the split, not the grader: training file and every holdout are
**length-balanced**, so the heuristic scores exactly 0.5000 by construction, and
`length_only` is now a permanent rung of the ladder so it can never quietly become the
ceiling again. Gate A then passes — the pretraining gain goes from +0.010 to +0.07…+0.17.

**One eval set of four clears the bar, and only just.** What stands in the way is how good
the *frozen* encoder is: a trained head on mean-pooled frozen embeddings gets to 0.6082
where a full fine-tune gets 0.6315. That is the protein task's finding again, in a new
domain. The task ships with `helpful_rs` at 3.10σ against a 3.0σ bar — honest, but thin,
and both its weaknesses are in [Open questions](#open-questions).

---

## What has actually been run

### `mol-property-adapt` — oracle through Harbor, Modal backend

`jobs/mol-oracle-modal/`, eval key `oracle__adhoc`, 1 trial, 0 errors.

| eval set | status | AUC | recovery | raw | test overlap | encoder tensors |
|---|---|---|---|---|---|---|
| `tox21` | ok | 0.6896 | 0.8193 | 0.8193 | 0 | 53 |
| `bbbp` | ok | 0.9158 | 1.0000 | **1.2575** | 0 | 53 |
| | | | **reward 0.909661** | | | |

- **`raw` 1.2575 on bbbp** — the oracle beat the reference by 26%. The capped `recovery`
  shows 1.0000 and hides that. Recording the uncapped value is how a mis-set anchor
  becomes visible instead of silently flattening to 1.0.
- **overlap 0** — the contamination check ran against a real Harbor artifact set,
  including `logs/agent/train_log.txt`, with no false positives.
- **tox21 at 0.6896** is *below* the minimum of the five seeds that set
  `reference_auc = 0.7019`. Unresolved — see [Open questions](#open-questions).

### `sciml-protein-regression` — verifier regrade

Both known artifacts, re-scored through the hardened grader on Linux:

| artifact | Spearman | reward | cosine_min | tensors |
|---|---|---|---|---|
| tier-0.5 lock | 0.43121570 | 0.5 | 0.99979 | 108 |
| **frozen probe** (no encoder adaptation) | 0.53577846 | **1.0** | 1.00000 | 108 |

Both reproduce their recorded values to the 7th decimal. The second row is the problem,
not a pass: a submission that never touches the encoder takes the maximum reward.

The oracle has since been run on GPU (`jobs/protein-oracle-gpu/`), which is what the task
was given a GPU for — its 4-epoch fine-tune could not finish inside its own budget on CPU.
It scored **Spearman 0.5733, reward 1.0**, at `cosine_min` 0.9838 over 108 tensors.

That is the one number that cuts *against* the shelving argument below: 0.5733 for a
fine-tune is above the 0.5358 the frozen-probe artifact scored on the same split. It does
not overturn the verdict, and it should not be read as if it does — the inversion was
measured over 13 frozen and 7 fine-tune seeds, and this is one seed of a different recipe.
What it does mean is that the ordering deserves a multi-seed re-measurement on GPU before
the task is written off for good. It also changes nothing about the *reward*, which is the
actual reason the task is shelved: three fixed tiers, both thresholds mis-set, and a
frozen probe still taking the top one.

### `qa-sft-adapt` — oracle through Harbor, Modal backend

`jobs/qa-sft-oracle-modal/`, eval key `oracle__adhoc`, 1 trial, 0 errors, 25m 16s total.

| eval set | status | accuracy | recovery | raw | shingle overlap |
|---|---|---|---|---|---|
| `arc_easy` | ok | 0.6967 | 1.0000 | 1.0104 | 0 |
| `sciq` | ok | 0.8367 | 1.0000 | 1.0738 | 0 |
| `openbookqa` | ok | 0.3700 | 1.0000 | 1.0848 | 0 |
| | | | **reward 1.0** | | |

The row that matters is `raw`, on all three: **1.01–1.08**. The shipped
`solution/train_reference.py`, run inside the agent container on hardware the anchors
were not measured on, reproduces the `reference_acc` it claims and edges slightly past
it. That is the invariant the mol task fails — its tox21 oracle lands *below* the anchor
its own docstring claims (Open question 1) — and it is why the claim is stated as a
number here rather than as prose.

The agent phase took 945 s of its 4-hour budget (3 epochs, best val_acc 0.7412 at epoch
2), leaving ~3.7 h for an agent to actually experiment. Shingle overlap was 0 against a
real Harbor artifact set including `logs/agent/train_log.txt` — no false positives.

### `pref-reward-model` — oracle through Harbor, Modal backend, GPU

`jobs/rm-oracle-modal/`, 1 trial, 0 errors.

| eval set | status | accuracy | recovery | raw | shingle overlap |
|---|---|---|---|---|---|
| `helpful_rs` | ok | 0.6189 | 0.5828 | 0.5828 | 0 |
| | | | **reward 0.582804** | | |

Agent phase: 514 s on GPU for 2 epochs over 38,420 pairs, best val pairwise accuracy
0.6296.

**Reward 0.58 for the oracle is the expected shape here, and the reason generalises.**
0.6189 is −1.29σ on the reference arm's seed spread and 0.0008 below the lowest of the
five seeds that set the anchor. `reference` is a five-seed **mean**, so a single-seed run
of the very same recipe lands below it roughly half the time by construction. An oracle
trial should be read as "recovery near, and often under, 1.0" — not as a pass/fail. To
get an oracle that reliably clears its own bar you must either anchor on a lower quantile
than the mean or run the oracle multi-seed, and both change what `reference` means.

This is the first run for which that distinction was measurable, and it is a partial
reframing of [Open question 1](#open-questions): mol's tox21 oracle is a *different*
problem, because it lands below the minimum of its seeds by far more than one σ.

**This run took two attempts, and the first failure is the useful part.** The task was
moved to a GPU after length-balancing made 8,000 pairs too few — but
`environment/Dockerfile` still installed torch from the CPU wheel index, and
`train_reference.py`, written when the task was CPU-only, never called `.to(device)`.
Either alone silently pins the reference to CPU at 2.85 pairs/s: **6.7 hours against a
4-hour timeout**, failing as a wall-clock timeout with no error, an hour after the
interesting part. No verifier suite can see this — it grades a checkpoint and never runs
the agent container — and no config review catches it, because `gpus = 1` is right and
the Dockerfile is right *for a different task*. **A task's compute class is not one
setting**, and the only thing that checks you changed all of it is an end-to-end run.

### The two post-training tasks — verifier regression suites

Not Harbor runs: these are `spike/posttrain/verify_graders.py`, which builds each real
verifier image and runs it under `--network none` against fixtures it constructs. The
accept path is asserted first.

| fixture | `qa-sft-adapt` | `pref-reward-model` |
|---|---|---|
| `base_unchanged` — provided base, fresh head | **accepted**, reward 0.0 (recovery_raw −0.014 / −0.364 / 0.0 recorded, not hidden) | **accepted**, reward 0.0 |
| `shuffled` — one embedding tensor permuted | rejected: `min per-tensor cosine 0.0002 < 0.9` | rejected: `min per-tensor cosine 0.0071 < 0.9` |
| `nan` — one tensor NaN'd | rejected: `non-finite weights in embed_tokens.weight` | rejected: `non-finite weights in embeddings.word_embeddings.weight` |
| `truncated` — config.json deleted | rejected: `missing config.json` | rejected: `missing config.json` |
| `public_twin` — bit-identical `all-distilroberta-v1` | n/a | rejected by **sha256**, the only layer that sees it |
| `laundered` — `SmolLM2-135M-Instruct` + 1e-4 noise | rejected by **nearest-ancestor** | n/a |
| `contaminated` — agent log quoting held-out rows | rejected: shingle overlap | rejected: shingle overlap |

Every case wrote a `reward.json`, which is the invariant that matters most: a missing
reward file is a trial error, not a zero.

---

## Why the protein task is shelved

Not compute — **ordering**. Across two splits, 13 frozen seeds and 7 fine-tune seeds at
std ~0.003:

| split | frozen probe | fine-tune | gap |
|---|---|---|---|
| shipped | 0.5494 ± 0.0033 | 0.5214 ± 0.0017 | −0.028 (−7.5σ) |
| MMseqs2 cluster, 30% id | 0.5784 ± 0.0029 | 0.5257 ± 0.0032 | −0.053 (−12.2σ) |

Fine-tuning reliably *loses*, so no threshold placement repairs it. Two independent
confirmations: [Kumar et al. 2022](https://arxiv.org/abs/2202.10054) (fine-tuning
distorts pretrained features under distribution shift) and
[iScience 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/), which finds
head-only *wins* on melting-point prediction specifically.

The cluster-split row is the counter-intuitive one: MMseqs2 cut prefix leakage from 89
sequences to 3 and the frozen probe got **better**. Assigning whole clusters to test
groups related proteins with similar Tm together, which is easier to rank.
**Removing near-duplicates and making a split harder are not the same thing.**

Its reward is also the wrong shape. It uses three fixed tiers rather than a continuous
recovery between measured anchors, and the repo proves both thresholds are mis-set:
`scripts/probe_ceiling.json` records `tiers_json_claims_frozen_probe: 0.3887` beside a
mean-pool Ridge at **0.4586**, a nonlinear head at **0.4973** and a trained head at
**0.546** — three frozen methods, none of which adapts the encoder, two of which clear
`t_strong = 0.45` outright. Quantization makes every calibration error maximal: a
submission 0.001 past a threshold scores identically to one 0.1 past, and a run near a
boundary flips tiers between reruns for a reward swing of 0.5. **Tiers look robust
because they hide small errors, and are fragile for exactly that reason.**

---

## What this repo learned about reward design

Recorded because ten eval sets across four tracks failed in different ways, and the
failures rhyme.

1. **Measure the whole ladder, not two anchors.** random-init → trivial heuristic →
   frozen probe → frozen + trained head → full adaptation. Check the ordering is
   monotone, that adjacent rungs are separated by ≫ noise, and that the top rung exceeds
   what a routine attempt reaches.
2. **`base` is the ceiling of the trivial class, not one member of it — and the trivial
   class includes things that are not models.** The protein task pinned its lower tier to
   a ridge probe at 0.3887 while a trained head reached 0.546, above its *top* tier. The
   same seam appeared on both mol eval sets and again on sciq. On the preference track
   the ceiling turned out to be a heuristic with no parameters at all: "pick the longer
   response" scored **0.6031** against a full fine-tune's 0.6042.
3. **Always run the random-init arm, and require its gap to clear noise.** An eval set can
   show a clean band while the pretrained weights contribute nothing, because both arms
   are learning the same surface feature. On the first preference cut a randomly
   initialized encoder scored 0.5938 against the pretrained fine-tune's 0.6042 — a gain
   smaller than the seed noise. `finalize_anchors.py` now refuses to ship an eval set
   whose pretraining gain is under 2σ.
4. **The size of the held-out set is a reward parameter, not a detail.** It sets the noise
   floor of every score the environment will ever produce. On the preference track the
   seed-to-seed spread at n=1,000 was ~0.015 against a binomial floor of 0.0156 — the
   noise was *measurement*, not training instability. Re-cutting the holdout at 4,000
   costs the verifier three minutes and buys back a factor of two in `band_sigma`.
5. **Anchors are functions of the split and the training size.** Re-measure after any
   change to either. Every failure here appeared *after* screening, when the split and
   train size were locked.
6. **Removing leakage ≠ increasing difficulty.** Both scientific tracks found the
   principled split made the task *easier* (protein above; BBBP scaffold shuffle
   0.726 → 0.921).
7. **Never clip silently.** Log the uncapped recovery. The clip is what hid a
   mis-calibrated reference for an entire working session.
8. **Fail closed on your own configuration.** Missing anchors, missing held-out keys or a
   missing sibling checkpoint are a broken image, not a default. Robustness to *agent*
   input and robustness to *your own* config are different problems.
9. **An anchor that is a mean is not a bar a single run clears.** `reference` is a
   five-seed mean, so the shipped oracle — the same recipe — lands below it about half
   the time by construction. `pref-reward-model`'s oracle returned recovery 0.5828 at
   −1.29σ, which is the distribution working, not a bug. Read an oracle trial as "near
   1.0, often under"; if you need it to clear reliably, anchor on a quantile instead and
   say so.
10. **A task's compute class is not one setting.** Moving `pref-reward-model` to a GPU
   meant `gpus = 1` in `task.toml` — and also the torch wheel index in the environment
   image, and a `.to(device)` in the reference script that was correct to omit when the
   task was CPU-only. Missing either pins the reference to CPU at 6.7 h against a 4 h
   timeout, and it fails as a silent wall-clock timeout. Only an end-to-end run finds it.
11. **Separate measuring from deciding.** `modal_measure.py` records what every arm
   scored; `finalize_anchors.py` turns that into anchors by stated rules. The rules can
   then be re-read and re-run in a second without a GPU — which is the difference between
   an anchor that is measured and one that was chosen once in a session nobody kept.

---

## Layout

| Path | What |
|---|---|
| [`common/`](common/) | verifier core, contamination matching, regrade driver, sync check |
| [`tasks/mol-property-adapt/`](tasks/mol-property-adapt/) | molecular property adaptation |
| [`tasks/qa-sft-adapt/`](tasks/qa-sft-adapt/) | supervised fine-tuning of a 135M causal LM |
| [`tasks/pref-reward-model/`](tasks/pref-reward-model/) | reward modelling on human preferences |
| [`sciml-protein-regression/`](sciml-protein-regression/) | shelved protein task; hardened verifier, measurement scripts |
| [`spike/`](spike/) | mol/protein split construction and anchors, [`SPIKE_RESULTS.md`](spike/SPIKE_RESULTS.md) |
| [`spike/posttrain/`](spike/posttrain/) | post-training corpora, splits, ladders, [`RESULTS.md`](spike/posttrain/RESULTS.md) |
| [`jobs/`](jobs/) | Harbor run outputs |

A Harbor task is two containers. `environment/` becomes the agent's (data, base model,
network); `tests/` becomes the verifier's (private split, anchors, grader, **no
network**). Only the paths listed in `task.toml`'s `artifacts` cross between them.

---

## Running things

The post-training task trees need their fixtures fetched before their verifier images can
build — 850 MB of pinned checkpoints that do not belong in git:

```bash
python spike/posttrain/assemble_tasks.py      # anchors, private rows, base + sibling models
python common/sync.py --check                 # shared grader modules have not drifted
```

Then:

```bash
# oracle trial through Harbor on Modal (local Docker on macOS silently
# ignores network_mode = "no-network", so the verifier would not be isolated)
harbor run -c tasks/mol-property-adapt/configs/job-modal.json --agent oracle
harbor run -c tasks/qa-sft-adapt/configs/job-modal.json --agent oracle
harbor run -c tasks/pref-reward-model/configs/job-modal.json --agent oracle

# re-score a finished trial without re-running the agent
./tasks/<task>/scripts/regrade.sh --all

# verifier regression suites, against the real images
python spike/posttrain/verify_graders.py                               # post-training tasks
modal run sciml-protein-regression/scripts/modal_verify_hardening.py   # 16 assertions
```

Both suites assert the *accept* path first — a grader that rejects honest submissions is
worse than the loopholes it closes, because a false reject is indistinguishable in the
reward from an agent that did nothing.

Re-deriving the post-training anchors from scratch is documented in
[`spike/posttrain/RESULTS.md`](spike/posttrain/RESULTS.md).

---

## Open questions

1. **tox21's oracle does not reproduce its own anchor.** 0.6896 against
   `reference_auc = 0.7019`, below the minimum of the 5 seeds that set it.
   `train_reference.py` and the anchor arm are the same recipe in two implementations.
   Either re-measure the anchor from the shipped script, or drop the claim in its
   docstring — the repo currently asserts both.
2. **`pref-reward-model` ships on one eval set at 3.10σ, chosen as the best of four.**
   3.10 against a 3.0 bar is a hair, so a third of that reward band is seed noise; and
   taking the maximum of four marginal measurements inflates the figure, so the honest
   reading is "around 3σ". It clears the bar this repo states and it is the weakest thing
   here. The obstacle is real rather than fixable by tuning: a trained head on frozen
   embeddings reaches 0.6082 where a full fine-tune reaches 0.6315.
3. **`base` is a max over noisy means, which biases it upward.** On `arc_easy` the two
   no-adaptation arms are 0.0013 apart with a σ of 0.0154, so which one "wins" is close to
   a coin flip. The bias is in the safe direction — it under-pays rather than pays for
   free — but a proper treatment would take an upper confidence bound.
4. **bbbp's contamination tripwire is thin.** Legitimate scores on that split run
   0.89–0.92, leaving ~0.02 below a fully test-trained model. The InChIKey overlap check
   carries most of that load.
5. **The agent gets one shot at a final artifact.** RE-Bench scores the best entry in an
   intermediate score log; that is not portable here, since it would require exposing the
   held-out set. So these tasks partly measure "did you select on validation" alongside
   "can you post-train a model."
6. **The protein task's GPU oracle beat its frozen probe** — Spearman 0.5733 against
   0.5358 on the same split — which is the opposite of the multi-seed result the shelving
   verdict rests on. One seed of one recipe is not enough to reopen it, but the ordering
   should be re-measured multi-seed on GPU rather than left as a table from the era when
   the oracle could not finish its own recipe. The reward shape is a separate and
   unaffected reason to shelve it.
7. **The shingle fingerprint is subsampled 1-in-4** to keep it under 4 MB inside the
   verifier image. A 30-token leak is still caught with probability 99.6%; a
   one-sentence leak is not certain to be.
8. **`MIN_ENCODER_TENSORS = 50` is still hardcoded in the mol grader.** The two new tasks
   take 90% of the base's own body-tensor count instead, which survives a model swap; the
   mol task was left alone so its recorded reward stays reproducible.

---

## Note on private holdout

This public repo includes the private seeds and the held-out rows by request. That
publishes the graded holdout; **do not treat the reward as un-gameable if agents can read
this repository.** The in-container defences — no verifier network, contamination
detection, and the implausible-score tripwire — assume the agent cannot see this repo.

---

## Regenerate data / caches

```bash
cd spike
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas scipy numpy scikit-learn torch \
    transformers safetensors huggingface_hub rdkit datasets
.venv/bin/python fetch_flip2.py            # FLIP2 pins
.venv/bin/python moleculenet.py --help     # MoleculeNet downloads
.venv/bin/python posttrain/fetch_data.py   # hh-rlhf + multiple-choice QA, with sha256 pins
```

Third-party clones (`.research/`, `_research/`), the spike venv, embedding caches, large
fixture checkpoints and the regenerable post-training corpora are gitignored.
