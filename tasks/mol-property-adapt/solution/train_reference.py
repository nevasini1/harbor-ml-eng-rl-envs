"""Reference fine-tune. Defines the upper anchor of the reward scale.

Recipe: discriminative learning rates (higher for the fresh head than the
pretrained encoder), one-cycle warmup and linear decay, masked BCE so missing
labels are excluded from the loss rather than treated as negatives, and epoch
selection on a validation slice held out of the agent's own training data.

The hyperparameters here match the `legal_finetune` arm of
scripts/modal_legal_anchors.py, which measured reference_auc: body LR 3e-5 rather
than 5e-5 and a 20% validation slice rather than 10%. reference_ablation.json puts
3e-5 at 0.7019 against 5e-5 at 0.7006 -- inside noise, so that was an alignment
change, not a tuning claim.

UNRESOLVED: this script does not currently reproduce the anchor it is supposed to
define. Run end to end on CPU it scored tox21 0.6897 (scripts/e2e_mol.json),
below the *minimum* of the five seeds that set reference_auc = 0.7019
(range 0.6967-0.7111). Same recipe, two implementations: this one shuffles
batches with torch.randperm and the anchor arm with rng.permutation, on different
hardware. bbbp landed on the other side, beating its own anchor at 0.9158 vs
0.9121, which is the tell -- the anchors track a GPU sibling of this script
rather than this script.

Two coherent resolutions, and the repo currently implies both:

  * treat reference_auc as "what this script produces" and re-measure it by
    running this file on CPU across ~5 seeds, so the oracle scores ~1.0 by
    construction; or
  * treat the reference as an independent tuned bar, in which case solve.sh is
    right that the oracle is "deliberately competent-but-ordinary ... so that a
    strong agent can exceed it", an oracle at 0.82 is working as intended, and
    this docstring should stop implying otherwise.

Until that is decided, do not assume editing this file moves the anchor.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = os.environ.get("BASE_MODEL_DIR", "/app/base_model")
DATA = Path(os.environ.get("DATA_DIR", "/app/data"))
OUT = Path(os.environ.get("OUTPUT_DIR", "/app/final_model"))
EVAL_SETS = ["tox21", "bbbp"]
EPOCHS = int(os.environ.get("EPOCHS", "20"))
SEED = 0


def mean_auc(y, p) -> float:
    aucs = []
    for t in range(y.shape[1]):
        m = ~np.isnan(y[:, t])
        if m.sum() and len(np.unique(y[m, t])) == 2:
            aucs.append(roc_auc_score(y[m, t], p[m, t]))
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def predict(model, tok, smiles, bs=64):
    model.eval()
    out = []
    for i in range(0, len(smiles), bs):
        enc = tok(smiles[i : i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
        out.append(model(**enc).logits.float().numpy())
    model.train()
    return np.concatenate(out)


def train(name: str) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    df = pd.read_csv(DATA / f"{name}_train.csv")
    labels = [c for c in df.columns if c != "smiles"]
    smiles = df["smiles"].tolist()
    Y = df[labels].to_numpy(dtype=np.float64)
    print(f"{name}: {len(df)} molecules, {len(labels)} tasks", flush=True)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(smiles))
    n_val = int(0.2 * len(order))
    val_idx, fit_idx = order[:n_val], order[n_val:]

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(labels), problem_type="multi_label_classification")

    yfit = torch.tensor(np.nan_to_num(Y[fit_idx], nan=0.0), dtype=torch.float32)
    wfit = torch.tensor(~np.isnan(Y[fit_idx]), dtype=torch.float32)
    fit_smiles = [smiles[i] for i in fit_idx]
    val_smiles = [smiles[i] for i in val_idx]

    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": 3e-5},
                             {"params": head, "lr": 1e-3}], weight_decay=0.01)
    bs = 32
    steps = EPOCHS * max(1, len(fit_smiles) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[3e-5, 1e-3], total_steps=steps, pct_start=0.1,
        anneal_strategy="linear")

    best_val, best_state, step = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(fit_smiles))
        for i in range(0, len(fit_smiles), bs):
            idx = perm[i : i + bs].tolist()
            enc = tok([fit_smiles[j] for j in idx], return_tensors="pt",
                      padding=True, truncation=True, max_length=256)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(**enc).logits, yfit[idx], weight=wfit[idx])
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step < steps:
                sched.step()
        val = mean_auc(Y[val_idx], predict(model, tok, val_smiles))
        print(f"  epoch {ep+1}/{EPOCHS} val_auc={val:.4f}", flush=True)
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tok.save_pretrained(dest)
    print(f"  saved {dest} (best val_auc={best_val:.4f})", flush=True)


if __name__ == "__main__":
    for ds in EVAL_SETS:
        train(ds)
