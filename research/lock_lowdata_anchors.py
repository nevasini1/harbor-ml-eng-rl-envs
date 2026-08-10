"""Lock the low-data private split (2000 train) and measure final anchors.

Widens the base→reference gap by giving the agent only 2000 labelled molecules
from the region-complement train pool. Screened gaps:
  N=500  → +0.042
  N=1000 → +0.011
  N=2000 → +0.098   ← selected
  N=4000 → +0.051
  full   → +0.023
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from make_private_split import inchikeys
from moleculenet import fit_predict, multitask_auc

HERE = Path(__file__).parent
BASE = "DeepChem/ChemBERTa-77M-MLM"
SEED = 0
N_TRAIN = 2000
EPOCHS = 20


@torch.no_grad()
def predict(model, tok, smiles, bs=64):
    model.eval()
    out = []
    for i in range(0, len(smiles), bs):
        enc = tok(smiles[i : i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
        out.append(torch.sigmoid(model(**enc).logits).numpy())
    return np.concatenate(out)


def embed(smiles, tok, model):
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), 64):
            enc = tok(list(smiles[i : i + 64]), return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
    return np.concatenate(out).astype(np.float32)


def finetune(tr, te, labels, tok, epochs, train_body, lr_body, lr_head, tag):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Ytr = tr[labels].to_numpy(dtype=np.float64)
    Yte = te[labels].to_numpy(dtype=np.float64)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(labels), problem_type="multi_label_classification")
    smiles = tr["smiles"].tolist()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(smiles))
    n_val = max(150, int(0.15 * len(order)))
    val_idx, fit_idx = order[:n_val], order[n_val:]
    yfit = torch.tensor(np.nan_to_num(Ytr[fit_idx], nan=0.0), dtype=torch.float32)
    wfit = torch.tensor(~np.isnan(Ytr[fit_idx]), dtype=torch.float32)
    fit_smiles = [smiles[i] for i in fit_idx]

    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    for p in body:
        p.requires_grad = train_body
    params = [{"params": head, "lr": lr_head}]
    if train_body:
        params.append({"params": body, "lr": lr_body})
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
        val = multitask_auc(Ytr[val_idx],
                            predict(model, tok, [smiles[i] for i in val_idx]))
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  [{tag}] ep {ep+1}/{epochs} val={val:.4f}", flush=True)

    model.load_state_dict(best_state)
    test = multitask_auc(Yte, predict(model, tok, te["smiles"].tolist()))
    print(f"  [{tag}] TEST={test:.4f} best_val={best_val:.4f}", flush=True)
    return float(test), float(best_val), model


def main() -> None:
    torch.set_num_threads(8)

    # Prefer the full region-complement pool if we already subsampled once.
    pool_path = HERE / "split" / "agent" / "tox21_train_pool_unused.csv"
    cur_path = HERE / "split" / "agent" / "tox21_train.csv"
    tr_full = pd.read_csv(pool_path if pool_path.exists() else cur_path)
    te = pd.read_csv(HERE / "split" / "private" / "tox21_test.csv")
    labels = [c for c in tr_full.columns if c != "smiles"]

    if len(tr_full) < N_TRAIN:
        raise SystemExit(f"train pool too small: {len(tr_full)} < {N_TRAIN}")

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(tr_full), size=N_TRAIN, replace=False))
    tr = tr_full.iloc[idx].reset_index(drop=True)

    (HERE / "split" / "agent").mkdir(parents=True, exist_ok=True)
    if not pool_path.exists():
        tr_full.to_csv(pool_path, index=False)
    tr.to_csv(cur_path, index=False)

    train_keys = set(k for k in inchikeys(tr["smiles"]) if k)
    overlap = sum(1 for k in inchikeys(te["smiles"]) if k and k in train_keys)
    print(f"locked train={len(tr)} test={len(te)} overlap={overlap}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    enc_model = AutoModel.from_pretrained(BASE)
    enc_model.eval()
    Etr = embed(tr["smiles"].tolist(), tok, enc_model)
    Ete = np.load(HERE / "cache" / "priv_tox21_test.npy")
    np.save(HERE / "cache" / "priv_tox21_train.npy", Etr)

    Ytr = tr[labels].to_numpy(dtype=np.float64)
    Yte = te[labels].to_numpy(dtype=np.float64)
    probe = multitask_auc(
        Yte, fit_predict(lambda: LogisticRegression(max_iter=3000), Etr, Ytr, Ete))
    print(f"frozen-probe={probe:.4f}", flush=True)

    head_auc, _, _ = finetune(tr, te, labels, tok, 5, False, 0.0, 1e-3, "head-only")
    ref_auc, ref_val, model = finetune(
        tr, te, labels, tok, EPOCHS, True, 5e-5, 1e-3, "oracle")
    print(f"\ngap_vs_probe={ref_auc - probe:+.4f}  "
          f"gap_vs_head={ref_auc - head_auc:+.4f}", flush=True)

    manifest_path = HERE / "split" / "private" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["datasets"]["tox21"].update({
        "n_train": len(tr),
        "n_test": len(te),
        "train_regime": "low-data subsample of region complement",
        "n_train_locked": N_TRAIN,
        "subsample_seed": SEED,
        "train_test_inchikey_overlap": overlap,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    anchors = {
        "tox21": {
            "n_train": len(tr),
            "n_test": len(te),
            "n_tasks": len(labels),
            "base_auc": round(float(probe), 4),
            "head_only_auc": round(head_auc, 4),
            "reference_auc": round(ref_auc, 4),
            "reference_val_auc": round(ref_val, 4),
            "gap": round(float(ref_auc - probe), 4),
            "gap_vs_head_only": round(float(ref_auc - head_auc), 4),
            "base_definition":
                "frozen backbone + logistic probe on mean-pooled embeddings",
            "reference_definition":
                "tuned fine-tune, 20 epochs, discriminative LRs, early stop on val",
        }
    }
    (HERE / "results" / "anchors_private.json").write_text(json.dumps(anchors, indent=2))
    print(json.dumps(anchors, indent=2), flush=True)

    dest = HERE / "fixtures" / "oracle" / "tox21"
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tok.save_pretrained(dest)
    print(f"saved oracle {dest}", flush=True)


if __name__ == "__main__":
    main()
