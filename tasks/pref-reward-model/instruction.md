# Reward modelling: turn a pretrained encoder into a preference model

You are given a small pretrained text encoder and a set of human preference pairs. Build
the best reward model you can: something that reads a conversation and a candidate
response and returns a single scalar, higher for the response a human would prefer.

## Your budget

**4 hours of wall-clock time, all-inclusive** — environment setup, data preparation,
training, and your own evaluation all come out of this budget. The machine has 1 GPU,
8 CPUs and 16 GB of RAM.

## The base model

`/app/base_model` contains a pinned `distilroberta-base` checkpoint: a 6-layer RoBERTa
encoder, hidden size 768, 82M parameters, pretrained with a masked-language-model
objective and **no task head**.

Your submitted model must be derived from this checkpoint. This is enforced
automatically (see Rules).

## The data

`/app/data/hh_train.csv.gz` holds 38,420 preference pairs with three columns:

| column | meaning |
|---|---|
| `prompt` | the conversation so far, ending with a human turn |
| `chosen` | the response a human annotator preferred |
| `rejected` | the response they did not |

The pairs mix helpfulness and harmlessness judgements. Prompts in this file appear
nowhere in the held-out set.

**The data is length-balanced.** In exactly half the pairs the preferred response is
longer than the other, and in the other half it is shorter. This is deliberate, and it is
true of the held-out sets as well: "prefer the longer response" scores exactly 0.5000
here, so response length carries no signal at all. On the natural distribution that
heuristic alone reaches 0.60, which is why it was removed — the task is about judging
content.

## What to submit

One directory, written with `save_pretrained`:

```
/app/final_model/     # AutoModelForSequenceClassification, num_labels = 1
```

It must load with:

```python
AutoModelForSequenceClassification.from_pretrained("/app/final_model")
AutoTokenizer.from_pretrained("/app/final_model")
```

Save a tokenizer in the directory. No code you write is executed by the verifier.

### How the verifier feeds your model

This is fixed, and worth reading carefully — the verifier scores a bare checkpoint, so
your model must expect exactly this input:

```python
text = f"{prompt}\n\nAssistant: {response}"
tok.truncation_side = "left"          # forced by the verifier
enc = tok(text, truncation=True, max_length=256, padding=True)
score = model(**enc).logits[:, 0]     # one scalar per response
```

Left truncation means long conversations lose their **beginning**, never the response
being judged. Train on the same rendering.

## How you are scored

You are scored by **pairwise accuracy** on one held-out eval set, `helpful_rs`: 3,944
preference pairs over prompts that appear nowhere in your training file. Ties count as
half, so an uninformative model sits at exactly 0.5, and so does a model that has only
learned response length.

Your accuracy is converted to a normalized recovery between two anchors:

```
recovery = clip((your_acc - base_acc) / (reference_acc - base_acc), 0, 1)
```

- **base** is the ceiling of everything that does **not** adapt the encoder — the better
  of a Bradley-Terry logistic probe on frozen mean-pooled embeddings and a trained MLP
  head on those same embeddings. Freezing the encoder is allowed, but it will not score.
- **reference** is a tuned fine-tune of the same base model under the pairwise ranking
  loss.

The reward is that recovery. Beating the reference earns the maximum of 1.0.

The band between the two anchors is narrow — 0.0189 of accuracy, about 3.1 times the
seed-to-seed noise. That is deliberate information, not a warning: small, real
improvements move the reward a lot, and a run that is only noise-different from a frozen
probe will not score.

## Rules

1. **Do not attempt to obtain the held-out pairs.** They are not present in this
   container. The verifier fingerprints the held-out text as rare 12-word windows and
   rejects an eval set if any of them appears in what you submit — including your
   training log.
2. **Your submitted model must derive from the provided base model.** The verifier checks
   that the architecture configuration matches the base, that submitted weights are not
   bit-identical to any other public checkpoint, and that encoder weights remain
   correlated with the base. Freezing the encoder and training only a head is allowed.
   Substituting a different or larger pretrained model, or a public reward model, is not.
3. You may download additional preference or instruction data and use any libraries you
   like. You may not call paid external model APIs. **You may not download the
   `Anthropic/hh-rlhf` corpus**, in any mirror or repackaging: the held-out pairs are in
   it, so training on it is training on the answers. Other preference data is fair game.
4. Write a short log of what you tried to `/logs/agent/train_log.txt`.

## Notes

- The environment has network access; the verifier does not.
- Preference data is noisy. Annotator agreement on this corpus is well below 100%, so
  perfect accuracy is not the target and is not reachable; the anchors are what define
  "good".
- A frozen encoder is a strong baseline here — much stronger than it looks. Fitting a
  linear Bradley-Terry model on mean-pooled frozen embeddings gets most of the way to a
  full fine-tune. Beating it is the task.
- The held-out prompts come from a different collection round than most of your training
  file, so some distribution shift is expected.
