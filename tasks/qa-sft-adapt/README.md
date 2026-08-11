# `qa-sft-adapt`

Supervised fine-tuning, the first stage of post-training, as an RL environment. The agent
gets `HuggingFaceTB/SmolLM2-135M` — a base model, never instruction-tuned — 3,986
multiple-choice science questions, and 4 hours on 8 CPUs. It must post-train the model so
it assigns higher likelihood to correct answers than to distractors.

**Status: active.** Strongest separation of anything in this repo.

## How it is scored

Nothing is generated and nothing is sampled. Each candidate answer is appended to a fixed
prompt and scored by the model's **mean log-probability per answer token**; the highest
scoring candidate is the prediction:

```python
prompt = f"Question: {question.strip()}\nAnswer:"
cont   = f" {choice.strip()}"          # scored, length-normalized, 160-token window
```

That makes the metric a deterministic function of the submitted weights. A
generation-plus-judge metric would put a second model's noise inside the reward.

```
recovery = clip((acc − base) / (reference − base), 0, 1)
reward   = integrity_gate × mean(recovery over the three eval sets)
```

## The measured ladder

5 seeds per arm, 600 held-out items per eval set.

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

`random_init` sits at chance (0.25 for four choices) everywhere, so the band is bought by
the pretrained weights rather than by the training loop.

**The ceiling arm differs by eval set**, which is why `base` is defined as a ceiling and
not as a named method: `head_only` — training only the tied output embedding and never the
transformer body — beats the untouched checkpoint by **+0.048** on sciq and *loses* to it
on openbookqa. Anchoring on "the base scored as-is" would have paid every head-only
submission ~37% of sciq's reward for no adaptation at all.

## Integrity

Four layers, failing on disjoint inputs: architecture-config hash, sha256 against pinned
public checkpoints, per-tensor float64 cosine against the base body, and
**nearest-ancestor**.

The fourth exists for this task specifically. `SmolLM2-135M-Instruct` is public and shares
this base's architecture exactly, so starting from it passes the config hash, passes the
sha256 check once the agent trains a step, and passes the cosine floor because the sibling
is itself derived from the base. Only comparing *which ancestor it is nearer to*
separates them. Verified: a perturbed instruct copy is rejected at mean cosine 1.000000 to
the sibling against 0.996951 to the base.

## Run it

```bash
harbor run -c tasks/qa-sft-adapt/configs/job-modal.json --agent oracle
./tasks/qa-sft-adapt/scripts/regrade.sh --all      # re-score without re-running the agent
```

The oracle scored **reward 1.0** through Harbor (`jobs/qa-sft-oracle-modal/`), at uncapped
recovery 1.0104 / 1.0738 / 1.0848 — the shipped `solution/` reproduces the `reference_acc`
it claims. Agent phase: 945 s of the 4-hour budget.

## An agent run

`codex` (gpt-5.6-sol) scored **reward 0.734115** in 2h 05m (`jobs/qa-sft-codex-modal/`):

| eval set | accuracy | base → reference | recovery |
|---|---|---|---|
| `arc_easy` | 0.6833 | 0.603 → 0.6957 | 0.867 |
| `sciq` | 0.8150 | 0.696 → 0.827 | 0.908 |
| `openbookqa` | 0.3367 | 0.315 → 0.3657 | **0.427** |

This is the task behaving as an evaluation should: it neither clipped to 1.0 nor floored to
0, and the shortfall is concentrated on the hardest eval set, where the base model is barely
above chance.

The agent also **out-designed the shipped oracle** without being told to. `train_reference.py`
holds back 398 items at random; the agent chose **598, source-stratified** — directly
addressing the selection noise that makes the oracle's best-epoch choice a 0.23σ coin flip.
It then ran a real ablation and rejected three ideas on evidence: distractor-ranking (a
0.0008 tie), MMLU science pre-training (hurt validation), and OpenBookQA-only continuation
(overfit).

It trained in **bf16**, which is worth noting: before `load_tensors` was switched to read
safetensors through torch, that submission would have crashed the grader with
`TypeError: data type 'bfloat16' not understood` rather than scoring anything.

Its holdout scores came in **below** its own local validation on two of three sets
(ARC 0.733 → 0.683, OBQA 0.416 → 0.337), so the private split is genuinely harder than a
local one.

Measurement and anchor derivation live in
[`research/posttrain/`](../../research/posttrain/RESULTS.md).
