"""Stronger reference recipe aimed at widening the base→reference gap.

Adds three things the screening recipe lacked:
  1. SMILES enumeration (RDKit) so each molecule is seen under multiple serializations
  2. Freeze-then-unfreeze: train the head alone for a few epochs, then unfreeze the body
  3. Longer schedule with early stopping on a train-internal validation slice

Used only after screen_gap.py picks a region. Writes the final anchors_private.json
and the oracle checkpoint the Harbor solve.sh will reproduce.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from moleculenet import morgan, multitask_auc, fit_predict

HERE = Path(__file__).parent
BASE = "DeepChem/ChemBERTa-77M-MLM"
SEED = 0


def enumerate_smiles(smiles: list[str], n: int, seed: int) -> list[str]:
    """Return n randomised SMILES per input (original always included once)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(seed)
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(smi)
            continue
        variants = {smi}
        for _ in range(n * 4):
            if len(variants) >= n:
                break
            variants.add(Chem.MolToSmiles(mol, doRandom=True))
        chosen = list(variants)[:n]
        while len(chosen) < n:
            chosen.append(smi)
        out.extend(chosen)
    return out


@torch.no_grad()
def predict(model, tok, smiles, bs=64):
    model.eval()
    out = []
    for i in range(0, len(smiles), bs):
        enc = tok(smiles[i : i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
        out.append(torch.sigmoid(model(**enc).logits).numpy())
    return np.concatenate(out)


def embed(smiles, threads: int, tag: str) -> np.ndarray:
    cache = HERE / "cache" / f"priv_{tag}.npy"
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


def train_strong(tr, te, labels, threads: int, head_epochs: int, full_epochs: int,
                 n_enum: int, lr: float, head_lr: float):
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(labels), problem_type="multi_label_classification")

    smiles = tr["smiles"].tolist()
    Y = tr[labels].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(smiles))
    n_val = max(300, int(0.1 * len(order)))
    val_idx, fit_idx = order[:n_val], order[n_val:]

    fit_smiles_raw = [smiles[i] for i in fit_idx]
    Yfit_raw = Y[fit_idx]
    # Enumerate: expand (smiles, labels) together.
    fit_smiles = enumerate_smiles(fit_smiles_raw, n_enum, SEED)
    Yfit = np.repeat(Yfit_raw, n_enum, axis=0)
    yfit = torch.tensor(np.nan_to_num(Yfit, nan=0.0), dtype=torch.float32)
    wfit = torch.tensor(~np.isnan(Yfit), dtype=torch.float32)

    val_smiles = [smiles[i] for i in val_idx]
    Yval = Y[val_idx]

    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]

    def run_phase(epochs, train_body: bool, phase_lr_body: float, phase_lr_head: float):
        nonlocal model
        for p in body:
            p.requires_grad = train_body
        params = [{"params": head, "lr": phase_lr_head}]
        if train_body:
            params.append({"params": body, "lr": phase_lr_body})
        opt = torch.optim.AdamW(params, weight_decay=0.01)
        bs = 32
        steps = epochs * max(1, len(fit_smiles) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[p["lr"] for p in params], total_steps=max(steps, 1),
            pct_start=0.1, anneal_strategy="linear")
        best_val, best_state, step = -1.0, None, 0
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
            val = multitask_auc(Yval, predict(model, tok, val_smiles))
            print(f"    ep {ep+1}/{epochs} val={val:.4f} "
                  f"(body={'on' if train_body else 'frozen'})", flush=True)
            if val > best_val:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        return best_val

    t0 = time.time()
    print(f"  phase 1: head-only ({head_epochs} ep, {n_enum}x SMILES enum)", flush=True)
    v1 = run_phase(head_epochs, train_body=False, phase_lr_body=0.0, phase_lr_head=head_lr)
    print(f"  phase 2: full fine-tune ({full_epochs} ep)", flush=True)
    v2 = run_phase(full_epochs, train_body=True, phase_lr_body=lr, phase_lr_head=head_lr)

    test = multitask_auc(te[labels].to_numpy(dtype=np.float64),
                         predict(model, tok, te["smiles"].tolist()))
    return float(test), float(max(v1, v2)), time.time() - t0, model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--full-epochs", type=int, default=25)
    ap.add_argument("--n-enum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--save-oracle", action="store_true")
    args = ap.parse_args()

    tr = pd.read_csv(HERE / "split" / "agent" / "tox21_train.csv")
    te = pd.read_csv(HERE / "split" / "private" / "tox21_test.csv")
    labels = [c for c in tr.columns if c != "smiles"]
    print(f"tox21: train={len(tr):,} test={len(te):,} tasks={len(labels)}")

    Etr = embed(tr["smiles"].tolist(), args.threads, "tox21_train")
    Ete = embed(te["smiles"].tolist(), args.threads, "tox21_test")
    Ytr = tr[labels].to_numpy(dtype=np.float64)
    Yte = te[labels].to_numpy(dtype=np.float64)
    base = multitask_auc(Yte, fit_predict(
        lambda: LogisticRegression(max_iter=3000), Etr, Ytr, Ete))
    print(f"  base (frozen probe)  AUC={base:.4f}", flush=True)

    print("  [strong oracle]")
    ref, val, secs, model, tok = train_strong(
        tr, te, labels, args.threads, args.head_epochs, args.full_epochs,
        args.n_enum, args.lr, args.head_lr)
    print(f"  reference (oracle)   AUC={ref:.4f}  (val {val:.4f}, {secs:.0f}s)",
          flush=True)
    print(f"  gap                  {ref - base:+.4f}")

    if args.save_oracle:
        dest = HERE / "fixtures" / "oracle" / "tox21"
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(dest)
        tok.save_pretrained(dest)
        print(f"  saved {dest}")

    out = {
        "tox21": {
            "n_train": len(tr), "n_test": len(te), "n_tasks": len(labels),
            "base_auc": round(base, 4),
            "reference_auc": round(ref, 4),
            "reference_val_auc": round(val, 4),
            "reference_seconds": round(secs, 1),
            "gap": round(ref - base, 4),
            "recipe": {
                "head_epochs": args.head_epochs,
                "full_epochs": args.full_epochs,
                "n_enum": args.n_enum,
                "lr": args.lr,
                "head_lr": args.head_lr,
            },
        }
    }
    dest = HERE / "results" / "anchors_private.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
