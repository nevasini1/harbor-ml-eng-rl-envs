"""Screen tox21 region holdouts for the LARGEST fine-tune - frozen-probe gap.

Selecting for lowest probe AUC alone gave us a thin +0.023 gap: the region was
hard for everyone, including the fine-tune. This screen runs a short fine-tune
on every candidate so we can pick on measured headroom.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from moleculenet import fetch, morgan, multitask_auc, fit_predict
from region_split import tanimoto_to, scoreable_tasks

HERE = Path(__file__).parent
BASE = "DeepChem/ChemBERTa-77M-MLM"
SEED = 0


def embed(smiles, threads: int, tag: str) -> np.ndarray:
    cache = HERE / "cache" / f"rs_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    cache.parent.mkdir(exist_ok=True)
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), 64):
            enc = tok(list(smiles[i : i + 64]), return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
    emb = np.concatenate(out).astype(np.float32)
    np.save(cache, emb)
    return emb


@torch.no_grad()
def predict(model, tok, smiles, bs=64):
    model.eval()
    out = []
    for i in range(0, len(smiles), bs):
        enc = tok(smiles[i : i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
        out.append(torch.sigmoid(model(**enc).logits).numpy())
    return np.concatenate(out)


def short_finetune(smiles_tr, Ytr, smiles_te, Yte, threads: int, epochs: int,
                   lr: float = 5e-5, head_lr: float = 1e-3):
    """Short reference recipe used only for screening (not the final oracle)."""
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=Ytr.shape[1], problem_type="multi_label_classification")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(smiles_tr))
    n_val = max(200, int(0.1 * len(order)))
    val_idx, fit_idx = order[:n_val], order[n_val:]

    yfit = torch.tensor(np.nan_to_num(Ytr[fit_idx], nan=0.0), dtype=torch.float32)
    wfit = torch.tensor(~np.isnan(Ytr[fit_idx]), dtype=torch.float32)
    fit_smiles = [smiles_tr[i] for i in fit_idx]

    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": lr},
                             {"params": head, "lr": head_lr}], weight_decay=0.01)
    bs = 32
    steps = epochs * max(1, len(fit_smiles) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr, head_lr], total_steps=steps, pct_start=0.1,
        anneal_strategy="linear")

    best_val, best_state, step = -1.0, None, 0
    t0 = time.time()
    for ep in range(epochs):
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
        val = multitask_auc(Ytr[val_idx],
                            predict(model, tok, [smiles_tr[i] for i in val_idx]))
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"      ep {ep+1}/{epochs} val={val:.4f}", flush=True)

    model.load_state_dict(best_state)
    test = multitask_auc(Yte, predict(model, tok, smiles_te))
    return float(test), float(best_val), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7731907)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--top-k-probe", type=int, default=8,
                    help="only fine-tune the K candidates with lowest probe AUC")
    args = ap.parse_args()

    df, labels, _ = fetch("tox21")
    smiles = df["smiles"].tolist()
    Y = df[labels].to_numpy(dtype=np.float64)
    F = (morgan(smiles) > 0).astype(np.float32)
    E = embed(smiles, args.threads, "tox21")
    n_test = int(args.test_frac * len(df))

    rng = np.random.default_rng(args.seed)
    candidates = rng.choice(len(df), size=args.candidates, replace=False)

    rows = []
    print(f"screening {len(candidates)} anchors (probe first) ...", flush=True)
    for anchor_i in candidates:
        sim = tanimoto_to(F, F[anchor_i])
        order = np.argsort(-sim)
        te, tr = np.sort(order[:n_test]), np.sort(order[n_test:])
        if scoreable_tasks(Y, te) < Y.shape[1] // 2:
            continue
        probe = multitask_auc(Y[te], fit_predict(
            lambda: LogisticRegression(max_iter=3000), E[tr], Y[tr], E[te]))
        rows.append({
            "anchor": int(anchor_i),
            "probe": round(float(probe), 4),
            "test_pos": round(float(np.nanmean(Y[te])), 4),
            "train_pos": round(float(np.nanmean(Y[tr])), 4),
            "mean_sim": round(float(sim[te].mean()), 4),
            "n_test": int(len(te)),
        })
        print(f"  anchor={anchor_i:>5} probe={probe:.4f}", flush=True)

    rows.sort(key=lambda r: r["probe"])
    shortlist = rows[: args.top_k_probe]
    print(f"\nfine-tuning top {len(shortlist)} by lowest probe ...", flush=True)

    for r in shortlist:
        sim = tanimoto_to(F, F[r["anchor"]])
        order = np.argsort(-sim)
        te, tr = np.sort(order[:n_test]), np.sort(order[n_test:])
        print(f"\n  [ft] anchor={r['anchor']}", flush=True)
        ft, val, secs = short_finetune(
            [smiles[i] for i in tr], Y[tr],
            [smiles[i] for i in te], Y[te],
            args.threads, args.epochs)
        r["finetune"] = round(ft, 4)
        r["finetune_val"] = round(val, 4)
        r["gap"] = round(ft - r["probe"], 4)
        r["ft_seconds"] = round(secs, 1)
        print(f"  -> probe={r['probe']:.4f} ft={ft:.4f} gap={r['gap']:+.4f} "
              f"({secs:.0f}s)", flush=True)

    shortlist.sort(key=lambda r: r.get("gap", -9), reverse=True)
    dest = HERE / "results" / "gap_screen.json"
    dest.write_text(json.dumps({"all_probes": rows, "shortlist": shortlist}, indent=2))

    print(f"\n{'anchor':>7} {'probe':>8} {'finetune':>9} {'gap':>8} {'val':>8}")
    for r in shortlist:
        print(f"{r['anchor']:>7} {r['probe']:>8.4f} {r.get('finetune', float('nan')):>9.4f} "
              f"{r.get('gap', float('nan')):>+8.4f} {r.get('finetune_val', float('nan')):>8.4f}")
    print(f"\nbest gap: anchor={shortlist[0]['anchor']} "
          f"gap={shortlist[0].get('gap'):+.4f}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
