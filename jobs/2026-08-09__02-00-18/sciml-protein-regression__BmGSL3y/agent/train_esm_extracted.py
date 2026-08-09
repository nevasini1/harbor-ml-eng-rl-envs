import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


BASE = "/models/esm2_t6_8M_UR50D"
DATA = "/data/train.csv.gz"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ProteinDataset(Dataset):
    def __init__(self, frame, mean=0.0, std=1.0):
        self.seq = frame.sequence.tolist()
        self.y = ((frame.target.to_numpy(np.float32) - mean) / std).astype(np.float32)
        self.length = frame.sequence.str.len().to_numpy()

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        return i, self.seq[i], self.y[i]


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
            chunk = order[start:start + self.bucket]
            chunk = chunk[np.argsort(self.lengths[chunk])]
            for j in range(0, len(chunk), self.batch_size):
                b = chunk[j:j + self.batch_size].tolist()
                if self.shuffle:
                    rng.shuffle(b)
                batches.append(b)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)


def make_collator(tokenizer):
    def collate(rows):
        idx, seq, y = zip(*rows)
        toks = tokenizer(list(seq), padding=True, return_tensors="pt")
        toks["labels"] = torch.tensor(y, dtype=torch.float32)
        toks["indices"] = torch.tensor(idx)
        return toks
    return collate


@torch.inference_mode()
def extract_split(model, loader, n):
    model.eval()
    out = np.empty((n, model.config.hidden_size), dtype=np.float32)
    started = time.time()
    for step, batch in enumerate(loader, 1):
        idx = batch.pop("indices").numpy()
        batch.pop("labels")
        z = model(**batch).last_hidden_state[:, 0].float().numpy()
        out[idx] = z
        if step % 100 == 0:
            print(f"extract {step}/{len(loader)} elapsed={time.time()-started:.1f}s", flush=True)
    return out


class Head(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x)).squeeze(-1)


def fit_head(xtr, ytr, xva, yva, seed=0):
    seed_all(seed)
    head = Head(xtr.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(xva)
    best = (-1, None, 0)
    for epoch in range(100):
        head.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), 256):
            ii = order[start:start + 256]
            pred = head(xt[ii])
            loss = nn.functional.smooth_l1_loss(pred, yt[ii], beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad(): pred = head(xv).numpy()
        rho = float(spearmanr(yva, pred).statistic)
        if rho > best[0]:
            best = (rho, {k: v.detach().clone() for k, v in head.state_dict().items()}, epoch)
        if epoch % 10 == 0:
            print(f"head epoch={epoch} rho={rho:.5f} best={best[0]:.5f}", flush=True)
        if epoch - best[2] > 20:
            break
    head.load_state_dict(best[1])
    print(f"head best rho={best[0]:.5f} epoch={best[2]}")
    return head


@torch.inference_mode()
def evaluate(model, loader, y_raw, label_mean, label_std):
    model.eval()
    preds = np.empty(len(y_raw), np.float32)
    losses = []
    for batch in loader:
        idx = batch.pop("indices").numpy()
        labels = batch.pop("labels")
        p = model(**batch).logits[:, 0]
        losses.append(nn.functional.mse_loss(p, labels).item() * len(labels))
        preds[idx] = p.numpy()
    rho = float(spearmanr(y_raw, preds).statistic)
    rmse = float(np.sqrt(np.mean((preds * label_std + label_mean - y_raw) ** 2)))
    return rho, rmse, sum(losses) / len(y_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", default="/app/checkpoints")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_all(args.seed)
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(DATA)
    trf = df[df.split.eq("train")].reset_index(drop=True)
    vaf = df[df.split.eq("val")].reset_index(drop=True)
    mean, std = float(trf.target.mean()), float(trf.target.std())
    tr = ProteinDataset(trf, mean, std)
    va = ProteinDataset(vaf, mean, std)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    collate = make_collator(tokenizer)
    tr_sampler = BucketBatchSampler(tr.length, args.batch_size, True, args.seed)
    va_sampler = BucketBatchSampler(va.length, args.batch_size * 2, False)
    tr_loader = DataLoader(tr, batch_sampler=tr_sampler, collate_fn=collate, num_workers=0)
    va_loader = DataLoader(va, batch_sampler=va_sampler, collate_fn=collate, num_workers=0)

    feature_file = "/app/esm_cls_features.npz"
    if args.extract or not os.path.exists(feature_file):
        backbone = AutoModel.from_pretrained(BASE)
        xtr = extract_split(backbone, tr_loader, len(tr))
        xva = extract_split(backbone, va_loader, len(va))
        np.savez_compressed(feature_file, xtr=xtr, xva=xva)
        del backbone
    else:
        z = np.load(feature_file); xtr, xva = z["xtr"], z["xva"]

    head = fit_head(xtr, tr.y, xva, va.y, args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
    model.classifier.load_state_dict(head.state_dict())
    del head, xtr, xva

    # Slightly smaller learning rate for pretrained layers than for the initialized task head.
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        (head_params if name.startswith("classifier.") else backbone_params).append(p)
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": head_params, "lr": args.lr * 3.0, "weight_decay": 0.01},
    ])
    total_updates = math.ceil(len(tr_loader) / args.accum) * args.epochs
    warmup = max(1, int(total_updates * 0.06))
    def lr_factor(step):
        if step < warmup: return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return max(0.08, 0.5 * (1.0 + math.cos(math.pi * progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    initial = evaluate(model, va_loader, vaf.target.to_numpy(), mean, std)
    print(f"initial rho={initial[0]:.5f} rmse={initial[1]:.3f}", flush=True)
    best_rho = initial[0]
    global_step = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train(); tr_sampler.set_epoch(epoch)
        opt.zero_grad(); run_loss = 0.0
        for step, batch in enumerate(tr_loader, 1):
            batch.pop("indices")
            labels = batch.pop("labels")
            pred = model(**batch).logits[:, 0]
            loss = nn.functional.smooth_l1_loss(pred, labels, beta=0.5) / args.accum
            loss.backward()
            run_loss += loss.item() * args.accum
            if step % args.accum == 0 or step == len(tr_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); global_step += 1
            if step % 200 == 0:
                print(f"epoch={epoch+1} step={step}/{len(tr_loader)} loss={run_loss/step:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2g} elapsed={time.time()-started:.0f}s", flush=True)
        rho, rmse, mse = evaluate(model, va_loader, vaf.target.to_numpy(), mean, std)
        print(f"EVAL epoch={epoch+1} rho={rho:.6f} rmse={rmse:.3f} norm_mse={mse:.4f}", flush=True)
        ep_dir = os.path.join(args.output, f"epoch_{epoch+1}")
        model.config.label_mean = mean
        model.config.label_std = std
        model.config.validation_spearman = rho
        model.save_pretrained(ep_dir)
        tokenizer.save_pretrained(ep_dir)
        with open(os.path.join(ep_dir, "metrics.json"), "w") as f:
            json.dump({"rho":rho, "rmse":rmse, "mean":mean, "std":std}, f)
        if rho > best_rho:
            best_rho = rho
            with open(os.path.join(args.output, "best.txt"), "w") as f: f.write(ep_dir)
    print(f"done best_rho={best_rho:.6f} elapsed={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()