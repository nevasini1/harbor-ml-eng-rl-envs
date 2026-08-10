# Molecular property prediction: adapt ChemBERTa under a CPU-only budget

You are given a small pretrained SMILES language model and labelled training data for
two molecular property prediction tasks. Produce the best possible predictors for both.

## Your budget

**4 hours of wall-clock time on CPU only, all-inclusive** — environment setup, data
preparation, training, and your own evaluation all come out of this budget. There is no
GPU. The machine has 8 CPUs and 16 GB of RAM.

## The base model

`/app/base_model` contains a pretrained ChemBERTa checkpoint
(`DeepChem/ChemBERTa-77M-MLM`): a RoBERTa encoder over SMILES strings, 3 layers,
hidden size 384, 3.4M parameters. It has been pretrained with a masked-language-model
objective and has **no task head**.

Your submitted models must be derived from this checkpoint. This is enforced
automatically (see Rules).

## The tasks

| eval set | task | training molecules | outputs |
|---|---|---|---|
| `bbbp` | blood-brain barrier penetration | 1,631 | 1 binary label |
| `tox21` | 12 toxicology assays | 2,000 | 12 binary labels, some missing |

Training data is at `/app/data/bbbp_train.csv` and `/app/data/tox21_train.csv`. Each has
a `smiles` column plus one column per label. **Missing labels are empty cells** and must
be excluded from the loss, not treated as negatives.

## What to submit

Two directories, each written with `save_pretrained`:

```
/app/final_model/bbbp/     # num_labels = 1
/app/final_model/tox21/    # num_labels = 12, in the column order of tox21_train.csv
```

Each must load with:

```python
AutoModelForSequenceClassification.from_pretrained(path)
AutoTokenizer.from_pretrained(path)
```

The verifier calls the model directly on SMILES strings and reads `logits`; higher
logits must mean higher probability of the positive class. Save a tokenizer in each
directory. No code you write is executed by the verifier.

## How you are scored

Each eval set is scored by **mean ROC-AUC across its tasks** on a held-out region of
chemical space: a contiguous neighbourhood of structurally similar molecules removed
whole from your training data. Expect a real distribution shift — the held-out molecules
are not a random sample of what you train on. Tasks with only one class present in the
held-out labels are skipped.

Your score on each eval set is converted to a normalized recovery between two anchors:

```
recovery = clip((your_auc - base_auc) / (reference_auc - base_auc), 0, 1)
```

- **base** is the provided model with the encoder frozen and only the standard
  sequence-classification head trained, early-stopped on a validation split — the best
  result reachable without adapting the encoder, and the floor you must beat to score
  anything. Freezing the encoder is allowed, but it will not score.
- **reference** is a tuned fine-tune of the same base model.

The final reward is the mean of the two recoveries. Beating the reference on both sets
earns the maximum of 1.0.

## Rules

1. **Do not attempt to obtain the held-out test molecules or labels.** The private split
   is not present in this container and is not derivable from the public MoleculeNet
   splits. The verifier computes the InChIKey of every molecule it can parse out of what
   you submit and rejects an eval set if any held-out molecule appears.
2. **Your submitted models must derive from the provided base model.** The verifier
   checks that the architecture configuration matches the base, that submitted weights
   are not bit-identical to any other public checkpoint, and that encoder weights remain
   correlated with the base. Freezing the encoder and training only a head is allowed.
   Substituting a different or larger pretrained model is not.
3. You may download additional training data and use any libraries you like. You may not
   call paid external model APIs. **You may not download the source MoleculeNet `bbbp` or
   `tox21` files**, in any mirror or repackaging: the held-out molecules and their labels
   are in them, so training on those files is training on the answers. Other chemical
   data is fair game.
4. Write a short log of what you tried to `/logs/agent/train_log.txt`.

## Notes

- The environment has network access; the verifier does not.
- Both datasets are class-imbalanced, and `tox21` has many missing labels.
- A full training run on `bbbp` takes a couple of minutes on this machine, and on
  `tox21` roughly ten. You have time for many experiments — use them.
