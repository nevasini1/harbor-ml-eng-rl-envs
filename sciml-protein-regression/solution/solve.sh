#!/bin/bash
# Oracle / reference: freeze ESM-2-8M backbone, train regression head on a
# CPU-friendly subsample. Prints are flushed so Harbor logs show progress.
set -euo pipefail

python -u - <<'PY'
import gzip
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "/models/esm2_t6_8M_UR50D"
DATA = "/data/train.csv.gz"
OUT = Path("/app/final_model")
MAX_LEN = 256
EPOCHS = 1
BATCH = 8
LR = 2e-4
THREADS = 4
MAX_TRAIN = 800  # fast oracle for smoke; agents may use the full train set

torch.set_num_threads(THREADS)
torch.manual_seed(0)
np.random.seed(0)

with gzip.open(DATA, "rt") as fh:
    df = pd.read_csv(fh)
train = df[df["split"] == "train"].reset_index(drop=True)
val = df[df["split"] == "val"].reset_index(drop=True)
if len(train) > MAX_TRAIN:
    train = train.sample(n=MAX_TRAIN, random_state=0).reset_index(drop=True)
print(f"oracle train={len(train)} val={len(val)} max_len={MAX_LEN}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
for p in model.esm.parameters():
    p.requires_grad = False
model.train()

class SeqDS(Dataset):
    def __init__(self, frame):
        self.seqs = frame["sequence"].astype(str).tolist()
        self.y = frame["target"].to_numpy(dtype=np.float32)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        return self.seqs[i], float(self.y[i])

def collate(batch):
    seqs, ys = zip(*batch)
    enc = tok(list(seqs), return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    enc["labels"] = torch.tensor(ys, dtype=torch.float32)
    return enc

train_loader = DataLoader(SeqDS(train), batch_size=BATCH, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
loss_fn = nn.MSELoss()
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    total = 0.0
    n = 0
    for step, batch in enumerate(train_loader, 1):
        labels = batch.pop("labels")
        opt.zero_grad(set_to_none=True)
        pred = model(**batch).logits.squeeze(-1)
        loss = loss_fn(pred, labels)
        loss.backward()
        opt.step()
        total += loss.item() * len(labels)
        n += len(labels)
        if step % 50 == 0:
            print(
                f"epoch {epoch} step {step}/{len(train_loader)} "
                f"mse={total/max(n,1):.4f} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    print(f"epoch {epoch}/{EPOCHS} train_mse={total/max(n,1):.4f} elapsed={time.time()-t0:.0f}s", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"wrote {OUT}", flush=True)
PY
