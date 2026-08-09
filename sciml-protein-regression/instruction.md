# Task: improve ESM-2-8M on Meltome thermostability prediction

You are given a pretrained protein language model and a training/validation table of protein sequences with melting-temperature labels. Improve the model. Your submission is scored automatically on a **private held-out test set you cannot see**.

## Budget

- **CPU only** (no GPU)
- **4 hours** wall-clock, all-inclusive (setup, data work, training, packaging)
- Network access is available in this environment for downloading training resources if you choose

## Provided assets

- Base model (pinned Hugging Face revision, already cached under `/models/esm2_t6_8M_UR50D`):
  - `facebook/esm2_t6_8M_UR50D`
- Labeled data at `/data/train.csv.gz` with columns:
  - `sequence` (amino-acid string, already truncated to length ≤ 512)
  - `target` (melting temperature in Celsius)
  - `split` ∈ {`train`, `val`}

You may use only the provided base model family as the starting checkpoint. Do **not** substitute a different public checkpoint.

## Submission contract

When finished, write a Hugging Face `save_pretrained` directory to:

```text
/app/final_model
```

Requirements:

1. Loadable with `AutoModelForSequenceClassification.from_pretrained("/app/final_model")`
2. `num_labels = 1` (regression head)
3. Same architecture as the provided ESM-2-8M base (anti-substitution checks will reject swapped models)
4. Tokenizer files should be present in `/app/final_model` as well

## Scoring

The verifier computes **Spearman correlation** between your model’s predictions and private ground-truth temperatures, then maps that to a scalar reward in `[0, 1]`. Higher is better. Malformed / unloadable / architecture-mismatched submissions score `0`.

## Rules

- Do not attempt to access, download, or reconstruct the private test set.
- Do not modify verifier code (you cannot see it anyway).
- Paid external model APIs are not allowed in the solution path.
- Objective is model quality on held-out ground truth, not pleasing a judge.
