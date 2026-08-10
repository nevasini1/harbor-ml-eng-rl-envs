# Supervised fine-tuning: post-train a 135M causal LM on science QA, on CPU

You are given a small pretrained causal language model and a set of multiple-choice
science questions with their answers. Post-train the model so that it assigns higher
likelihood to correct answers than to distractors.

## Your budget

**4 hours of wall-clock time on CPU only, all-inclusive** — environment setup, data
preparation, training, and your own evaluation all come out of this budget. There is no
GPU. The machine has 8 CPUs and 16 GB of RAM.

## The base model

`/app/base_model` contains a pinned `HuggingFaceTB/SmolLM2-135M` checkpoint: a 30-layer
Llama-architecture decoder, hidden size 576, 135M parameters. It is a **base** model —
pretrained on text, never instruction-tuned.

Your submitted model must be derived from this checkpoint. This is enforced
automatically, and the enforcement specifically covers the instruction-tuned sibling of
this model (see Rules).

## The data

`/app/data/qa_train.csv` holds 3,986 multiple-choice items drawn from the same three
corpora as the eval sets:

| column | meaning |
|---|---|
| `question` | the question stem |
| `choices` | a JSON list of candidate answers |
| `answer_idx` | index of the correct one |
| `source` | which corpus the item came from |

## What to submit

One directory, written with `save_pretrained`:

```
/app/final_model/     # AutoModelForCausalLM
```

It must load with:

```python
AutoModelForCausalLM.from_pretrained("/app/final_model")
AutoTokenizer.from_pretrained("/app/final_model")
```

Save a tokenizer in the directory. No code you write is executed by the verifier.

### How the verifier scores your model

Nothing is generated and nothing is sampled. For each item, every candidate answer is
appended to a fixed prompt and scored by the model's **mean log-probability per answer
token**; the highest-scoring candidate is the prediction:

```python
prompt = f"Question: {question.strip()}\nAnswer:"
cont   = f" {choice.strip()}"
# mean log P(cont tokens | prompt), sequence truncated to 160 tokens from the left
```

Length normalization is by token count. Train on exactly this rendering — a model tuned
for a different prompt format will be scored on this one anyway.

## How you are scored

There are three eval sets, one per source corpus, each 600 held-out items your training
file does not contain:

| eval set | what it is | how hard it is for the base model |
|---|---|---|
| `arc_easy` | grade-school science exam questions | moderate |
| `sciq` | crowdsourced science questions | easiest |
| `openbookqa` | science questions needing a retrieved fact | hardest — the base model is barely above chance |

Each is scored by **accuracy** under the ranking above. Your accuracy is converted to a
normalized recovery between two anchors:

```
recovery = clip((your_acc - base_acc) / (reference_acc - base_acc), 0, 1)
```

- **base** is the ceiling of everything that does **not** adapt the transformer body —
  the better of the provided checkpoint scored as-is and a run that trains only the
  output embedding matrix. Submitting the base unchanged scores 0.
- **reference** is a tuned supervised fine-tune of the same base model.

The final reward is the mean of the recoveries. Beating the reference on every eval set
earns the maximum of 1.0.

## Rules

1. **Do not attempt to obtain the held-out items.** They are not present in this
   container. The verifier fingerprints the held-out text as rare 12-word windows and
   rejects an eval set if any of them appears in what you submit — including your
   training log.
2. **Your submitted model must derive from the provided base model.** The verifier checks
   that the architecture configuration matches the base, that submitted weights are not
   bit-identical to any other public checkpoint, that weights remain correlated with the
   base, **and that they are closer to the base than to any same-architecture public
   sibling**. In particular, starting from `SmolLM2-135M-Instruct` and fine-tuning it is
   detected and rejected: it has an identical config, so only an ancestry comparison
   separates it from an honest run. Start from the checkpoint you were given.
3. You may download additional QA or instruction data and use any libraries you like. You
   may not call paid external model APIs. **You may not download `allenai/ai2_arc`,
   `allenai/sciq` or `allenai/openbookqa`**, in any mirror or repackaging: the held-out
   items are in them. Other data is fair game.
4. Write a short log of what you tried to `/logs/agent/train_log.txt`.

## Notes

- The environment has network access; the verifier does not.
- A 3-epoch fine-tune over the full training file takes about 20 minutes on this machine
  (measured: 1.5 s per step of 16 items, 672 steps, plus validation passes). You have
  time for many experiments — use them.
- The base model is small and is not instruction-tuned. Its zero-shot accuracy under the
  scoring above is well below what the same model reaches after fine-tuning, which is the
  headroom you are being scored on.
- Training on the correct answer only is the obvious recipe. It is not the only one — the
  distractors are labelled data too.
