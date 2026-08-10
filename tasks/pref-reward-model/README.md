# `pref-reward-model`

Reward modelling — RLHF stage 2 — as an RL environment. The agent gets
`distilroberta-base`, 38,420 human preference pairs from hh-rlhf, one GPU and 4 hours. It
must post-train the encoder into something that reads a conversation and a candidate
response and returns a scalar, higher for the response a human preferred.

**Status: active, and the weakest thing in this repo.** It ships on one eval set at 3.10σ
against a 3.0σ bar. Read [Honest limits](#honest-limits) before using it.

## How it is scored

One forward pass per response, no generation:

```python
text = f"{prompt}\n\nAssistant: {response}"
tok.truncation_side = "left"          # forced by the verifier
score = model(**enc).logits[:, 0]     # num_labels = 1
```

Left truncation is load-bearing: long conversations lose their beginning, never the
response being judged. Metric is **pairwise accuracy with ties counted as half** — the
only convention that leaves an uninformative model at exactly chance.

## The measured ladder

5 seeds per arm, ~3,960 held-out pairs per eval set, on the length-balanced split.

| arm | helpful_base | helpful_rs | online | harmless |
|---|---|---|---|---|
| `length_only` (pick the longer response) | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `frozen_probe` (Bradley-Terry on frozen embeddings) | 0.6055 | 0.5976 | 0.5496 | 0.5696 |
| `frozen_head` (trained MLP on those embeddings) | 0.6082 ± 0.0056 | 0.6079 ± 0.0046 | 0.5610 ± 0.0042 | 0.6219 ± 0.0101 |
| `finetune` — **reference** | 0.6315 ± 0.0087 | 0.6268 ± 0.0061 | 0.5633 ± 0.0057 | 0.6390 ± 0.0064 |
| `random_init` | 0.5496 ± 0.0061 | 0.5525 ± 0.0102 | 0.5174 ± 0.0040 | 0.4649 ± 0.0102 |

Only `helpful_rs` ships, at band 0.0189 and **3.10σ**.

## Why the data is length-balanced

The first cut of this task measured **response length and nothing else**. On a natural
sample of hh-rlhf, "pick the longer response" — no model, no parameters — scored
**0.6031** against a full fine-tune's 0.6042 and a frozen-encoder ceiling of 0.5646. That
heuristic alone covered 97% of the reward band, and a randomly-initialized encoder reached
0.5938, inside the seed noise of the pretrained one.

The tell was `harmless`: every model arm scored *below chance* there, because they had
learned "longer is better" from a training file where it held (0.5356) and it is backwards
on that split (0.4116). They were length detectors with a sign error, not preference
models.

So the split — training file and every holdout — is length-balanced: equal counts of
longer-chosen and shorter-chosen pairs, ties dropped. `length_only` now reads exactly
0.5000 by construction and is a permanent rung of the ladder, so the heuristic can never
quietly become the ceiling again. Pretraining gain went from +0.010 to +0.07…+0.17.

Balancing is applied to the *training* file too. Leaving it biased would teach every
submission a feature worth nothing at grade time, which measures whether the agent spotted
a trap rather than whether it can post-train.

## Why it needs a GPU

Not a compute complaint — a consequence of the fix above. Length-balancing removed the
shortcut, and a model that has to judge content needs far more data: the frozen probe
alone moves 0.5778 → 0.6055 between 8,000 and 38,420 pairs. Two epochs over 38,420 pairs
is 6.7 hours on 8 CPU cores at the measured 2.85 pairs/s, so CPU-only would have meant
keeping the task at a scale where nothing separates the arms.

## Honest limits

- **3.10σ against a 3.0σ bar is a hair.** A third of the reward band is seed noise. For
  scale, `mol-property-adapt`'s `bbbp` ships at 4.09σ.
- **It is the best of four screened eval sets.** Taking the maximum of four marginal
  measurements inflates it; read it as "around 3σ".
- **The obstacle is real, not a tuning failure.** A trained head on frozen embeddings
  reaches 0.6082 where a full fine-tune reaches 0.6315. A 3-epoch reference was measured
  and *rejected* — it made things worse (0.6255 ± 0.0104).

## Run it

```bash
harbor run -c tasks/pref-reward-model/configs/job-modal.json --agent oracle
./tasks/pref-reward-model/scripts/regrade.sh --all
```

The oracle scored **reward 0.5828** through Harbor (`jobs/rm-oracle-modal/`). That is the
expected shape, not a defect: 0.6189 is −1.29σ on the reference arm's seed spread, and
`reference` is a five-seed *mean*, which a single seed lands under about half the time.

Measurement and anchor derivation live in
[`research/posttrain/`](../../research/posttrain/RESULTS.md).
