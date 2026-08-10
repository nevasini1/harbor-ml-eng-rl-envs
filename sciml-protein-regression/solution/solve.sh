#!/bin/bash
# Strong oracle:
# 1) stream frozen CLS features
# 2) fit a small head on them
# 3) full-model fine-tune, warm-started from that head
#
# Runs on GPU when one is present and falls back to CPU otherwise. The fallback
# is not practical -- a single forward pass over the corpus is ~85 min at 512
# tokens and this does four fine-tune epochs -- but it keeps the script runnable
# for debugging without a GPU.
set -euo pipefail

python -u - <<'PY'
import gc
import gzip
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

BASE = "/models/esm2_t6_8M_UR50D"
DATA = "/data/train.csv.gz"
OUT = Path("/app/final_model")
FEAT = Path("/tmp/oracle_cls_features.npz")
MAX_LEN = 512
EPOCHS = 4
BATCH = 2
ACCUM = 4
LR = 7e-5
THREADS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    # Keep the optimization identical to the CPU path: BATCH*ACCUM stays 8, and
    # the number of optimizer updates per epoch is unchanged, so LR and the
    # warmup/cosine schedule carry over. Only the micro-batch grows, because
    # gradient accumulation existed solely to fit CPU memory.
    BATCH, ACCUM = 8, 1
print(f"device={DEVICE} batch={BATCH} accum={ACCUM}", flush=True)

torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)
torch.manual_seed(17)
np.random.seed(17)

with gzip.open(DATA, "rt") as fh:
    df = pd.read_csv(fh)
train = df[df["split"] == "train"].reset_index(drop=True)
val = df[df["split"] == "val"].reset_index(drop=True)

mean = float(train["target"].mean())
std = max(float(train["target"].std()), 1e-6)
print(f"oracle train={len(train)} val={len(val)} mean={mean:.3f} std={std:.3f}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE)


class SeqDS(Dataset):
    def __init__(self, frame, mean, std):
        self.seqs = frame["sequence"].astype(str).tolist()
        self.y = ((frame["target"].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
        self.y_raw = frame["target"].to_numpy(dtype=np.float32)
        self.lengths = np.array([min(len(s), MAX_LEN) for s in self.seqs], dtype=np.int32)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return i, self.seqs[i], float(self.y[i])


class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle, seed=0, bucket_mult=40):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.bucket = batch_size * bucket_mult

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(len(self.lengths))
        if self.shuffle:
            rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.bucket):
            chunk = order[start : start + self.bucket]
            chunk = chunk[np.argsort(self.lengths[chunk])]
            for j in range(0, len(chunk), self.batch_size):
                b = chunk[j : j + self.batch_size].tolist()
                if self.shuffle:
                    rng.shuffle(b)
                batches.append(b)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)


def collate(batch):
    idx, seqs, ys = zip(*batch)
    enc = tok(list(seqs), return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    enc["labels"] = torch.tensor(ys, dtype=torch.float32)
    enc["indices"] = torch.tensor(idx, dtype=torch.long)
    return enc


tr_ds = SeqDS(train, mean, std)
va_ds = SeqDS(val, mean, std)


@torch.no_grad()
def extract_cls(model, loader, n):
    model.eval()
    xs = np.empty((n, model.config.hidden_size), dtype=np.float32)
    ys = np.empty(n, dtype=np.float32)
    for step, batch in enumerate(loader, 1):
        idx = batch.pop("indices").numpy()
        labels = batch.pop("labels")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        h = model(**batch).last_hidden_state[:, 0].float().cpu().numpy()
        xs[idx] = h
        ys[idx] = labels.numpy()
        if step % 200 == 0:
            print(f"extract step={step}/{len(loader)}", flush=True)
    return xs, ys


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x)).squeeze(-1)


def fit_head(xtr, ytr, xva, yva):
    head = Head(xtr.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xv = torch.from_numpy(xva)
    best_rho, best_state, best_ep = -1.0, None, 0
    for epoch in range(100):
        head.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), 256):
            ii = order[start : start + 256]
            loss = nn.functional.smooth_l1_loss(head(xt[ii]), yt[ii], beta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(xv).numpy()
        rho = float(spearmanr(yva, pred).statistic)
        if rho > best_rho:
            best_rho, best_state, best_ep = rho, {k: v.detach().clone() for k, v in head.state_dict().items()}, epoch
        if epoch % 10 == 0:
            print(f"head epoch={epoch} rho={rho:.4f} best={best_rho:.4f}", flush=True)
        if epoch - best_ep > 20:
            break
    head.load_state_dict(best_state)
    print(f"head best rho={best_rho:.4f} epoch={best_ep}", flush=True)
    return head


@torch.no_grad()
def eval_rho(model, loader, y_raw):
    model.eval()
    preds = np.empty(len(y_raw), dtype=np.float32)
    for batch in loader:
        idx = batch.pop("indices").numpy()
        batch.pop("labels")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        preds[idx] = model(**batch).logits.squeeze(-1).float().cpu().numpy()
    pred = preds * std + mean
    return float(spearmanr(y_raw, pred).statistic)


tr_sampler_ext = BucketBatchSampler(tr_ds.lengths, BATCH, False, seed=17)
va_sampler_ext = BucketBatchSampler(va_ds.lengths, BATCH * 2, False, seed=18)
tr_loader_ext = DataLoader(tr_ds, batch_sampler=tr_sampler_ext, collate_fn=collate)
va_loader_ext = DataLoader(va_ds, batch_sampler=va_sampler_ext, collate_fn=collate)

if FEAT.exists():
    z = np.load(FEAT)
    xtr, ytr, xva, yva = z["xtr"], z["ytr"], z["xva"], z["yva"]
    print(f"loaded features {FEAT} tr={xtr.shape}", flush=True)
else:
    print("extracting frozen CLS features...", flush=True)
    backbone = AutoModel.from_pretrained(BASE).to(DEVICE)
    xtr, ytr = extract_cls(backbone, tr_loader_ext, len(tr_ds))
    xva, yva = extract_cls(backbone, va_loader_ext, len(va_ds))
    del backbone
    gc.collect()
    np.savez_compressed(FEAT, xtr=xtr, ytr=ytr, xva=xva, yva=yva)
    print(f"features ready tr={xtr.shape} va={xva.shape}", flush=True)

head = fit_head(xtr, ytr, xva, yva)
del xtr, ytr, xva, yva
gc.collect()

model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEVICE)
model.classifier.load_state_dict(head.state_dict())
del head
gc.collect()

tr_sampler = BucketBatchSampler(tr_ds.lengths, BATCH, True, seed=17)
va_sampler = BucketBatchSampler(va_ds.lengths, BATCH * 2, False, seed=18)
tr_loader = DataLoader(tr_ds, batch_sampler=tr_sampler, collate_fn=collate)
va_loader = DataLoader(va_ds, batch_sampler=va_sampler, collate_fn=collate)

backbone_params, head_params = [], []
for name, p in model.named_parameters():
    (head_params if name.startswith("classifier.") else backbone_params).append(p)
print(f"trainable tensors={sum(1 for _ in model.parameters())}", flush=True)
opt = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": LR, "weight_decay": 0.01},
        {"params": head_params, "lr": LR * 3.0, "weight_decay": 0.01},
    ]
)
total_updates = math.ceil(len(tr_loader) / ACCUM) * EPOCHS
warmup = max(1, int(total_updates * 0.06))


def lr_factor(step):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_updates - warmup)
    return max(0.08, 0.5 * (1.0 + math.cos(math.pi * progress)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
print(f"initial val spearman={eval_rho(model, va_loader, va_ds.y_raw):.4f}", flush=True)

t0 = time.time()
best_rho = -1.0
best_state = None
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_sampler.set_epoch(epoch)
    opt.zero_grad(set_to_none=True)
    total = 0.0
    n = 0
    for step, batch in enumerate(tr_loader, 1):
        batch.pop("indices")
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        pred = model(**batch).logits.squeeze(-1)
        loss = nn.functional.smooth_l1_loss(pred, labels, beta=0.5) / ACCUM
        loss.backward()
        total += float(loss.item() * ACCUM) * len(labels)
        n += len(labels)
        if step % ACCUM == 0 or step == len(tr_loader):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if step % 200 == 0:
            print(
                f"epoch {epoch} step {step}/{len(tr_loader)} "
                f"loss={total/max(n,1):.4f} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    rho = eval_rho(model, va_loader, va_ds.y_raw)
    print(f"epoch {epoch}/{EPOCHS} val_spearman={rho:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
    if rho > best_rho:
        best_rho = rho
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)
model = model.cpu()
model.config.label_mean = mean
model.config.label_std = std
model.config.validation_spearman = best_rho
OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"wrote {OUT} best_val_spearman={best_rho:.4f}", flush=True)
PY
