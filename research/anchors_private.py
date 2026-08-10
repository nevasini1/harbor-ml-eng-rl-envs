"""Establish the verifier's anchor values on the private split.

Anchors:
  base      - frozen backbone + logistic probe. The cheapest legitimate use of the
              provided model: no backbone training, ~30 seconds of work. This is the
              floor the agent must beat to earn any reward.
  reference - the tuned fine-tune recipe that becomes solution/solve.sh.

An untrained-head model would score ~0.5 AUC, which would make the floor trivially
easy to clear and waste most of the reward range on work that requires no
engineering. Anchoring at the frozen probe spends the whole 0-1 range on the part
of the problem we actually want to measure.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from moleculenet import morgan, multitask_auc, fit_predict

HERE = Path(__file__).parent
SPLIT = HERE / "split"
BASE_MODEL = "DeepChem/ChemBERTa-77M-MLM"
EVAL_SETS = {"tox21": None}
SEED = 0


def load(name: str):
    tr = pd.read_csv(SPLIT / "agent" / f"{name}_train.csv")
    te = pd.read_csv(SPLIT / "private" / f"{name}_test.csv")
    labels = [c for c in tr.columns if c != "smiles"]
    return tr, te, labels


def embed(smiles, threads: int, tag: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache = HERE / "cache" / f"priv_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    cache.parent.mkdir(exist_ok=True)
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL)
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


def oracle_finetune(tr, te, labels, threads: int, epochs: int, seed: int,
                    lr: float = 5e-5, head_lr: float = 1e-3, val_frac: float = 0.1):
    """Reference recipe: warmup + linear decay, separate head LR, early stopping on a
    validation slice carved out of the agent's own training data."""
    import torch
    from transformers import AutoTokenizer, RobertaForSequenceClassification

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = RobertaForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels),
        problem_type="multi_label_classification")

    smiles = tr["smiles"].tolist()
    Y = tr[labels].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(smiles))
    n_val = int(val_frac * len(order))
    val_idx, fit_idx = order[:n_val], order[n_val:]

    yfit = torch.tensor(np.nan_to_num(Y[fit_idx], nan=0.0), dtype=torch.float32)
    wfit = torch.tensor(~np.isnan(Y[fit_idx]), dtype=torch.float32)
    fit_smiles = [smiles[i] for i in fit_idx]

    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": lr},
                             {"params": head, "lr": head_lr}], weight_decay=0.01)
    bs = 32
    steps = epochs * max(1, len(fit_smiles) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr, head_lr], total_steps=steps, pct_start=0.1, anneal_strategy="linear")

    best_val, best_state, best_ep = -1.0, None, -1
    t0 = time.time()
    step = 0
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
        val_auc = multitask_auc(Y[val_idx],
                                predict(model, tok, [smiles[i] for i in val_idx]))
        if val_auc > best_val:
            best_val, best_ep = val_auc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    ep {ep+1}/{epochs} val={val_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    test_auc = multitask_auc(te[labels].to_numpy(dtype=np.float64),
                             predict(model, tok, te["smiles"].tolist()))
    return test_auc, best_val, best_ep + 1, time.time() - t0, model, tok


def predict(model, tok, smiles, bs: int = 64):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), bs):
            enc = tok(smiles[i : i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            out.append(torch.sigmoid(model(**enc).logits).numpy())
    model.train()
    return np.concatenate(out)


def main() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--save-oracle", action="store_true")
    args = ap.parse_args()

    out = {}
    for name in EVAL_SETS:
        tr, te, labels = load(name)
        Ytr = tr[labels].to_numpy(dtype=np.float64)
        Yte = te[labels].to_numpy(dtype=np.float64)
        print(f"\n{'='*72}\n{name}: train={len(tr):,} test={len(te):,} tasks={len(labels)}")

        Etr = embed(tr["smiles"].tolist(), args.threads, f"{name}_train")
        Ete = embed(te["smiles"].tolist(), args.threads, f"{name}_test")
        t0 = time.time()
        base = multitask_auc(Yte, fit_predict(
            lambda: LogisticRegression(max_iter=3000), Etr, Ytr, Ete))
        print(f"  base (frozen probe)  AUC={base:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        Ftr, Fte = morgan(tr["smiles"].tolist()), morgan(te["smiles"].tolist())
        rf = multitask_auc(Yte, fit_predict(
            lambda: RandomForestClassifier(n_estimators=300, n_jobs=args.threads,
                                           random_state=SEED), Ftr, Ytr, Fte))
        print(f"  morgan+rf            AUC={rf:.4f}", flush=True)

        print("  [oracle fine-tune]")
        ref, val, ep, secs, model, tok = oracle_finetune(
            tr, te, labels, args.threads, args.epochs, SEED)
        print(f"  reference (oracle)   AUC={ref:.4f}  "
              f"(best epoch {ep}, val {val:.4f}, {secs:.0f}s)", flush=True)

        if args.save_oracle:
            dest = HERE / "fixtures" / "oracle" / name
            dest.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(dest)
            tok.save_pretrained(dest)
            print(f"  saved oracle checkpoint -> {dest}")

        out[name] = {
            "n_train": len(tr), "n_test": len(te), "n_tasks": len(labels),
            "base_auc": round(base, 4), "morgan_rf_auc": round(rf, 4),
            "reference_auc": round(ref, 4), "reference_best_epoch": ep,
            "reference_val_auc": round(val, 4), "reference_seconds": round(secs, 1),
            "gap": round(ref - base, 4),
        }

    dest = HERE / "results" / "anchors_private.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n{'dataset':<8} {'base':>8} {'morgan-rf':>10} {'reference':>10} {'gap':>8}")
    for k, v in out.items():
        print(f"{k:<8} {v['base_auc']:>8.4f} {v['morgan_rf_auc']:>10.4f} "
              f"{v['reference_auc']:>10.4f} {v['gap']:>+8.4f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
