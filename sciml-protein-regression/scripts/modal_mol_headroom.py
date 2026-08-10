"""Does mol-property-adapt have the headroom the protein task lacks?

Why
---
The protein task is inverted: across two splits, 13 frozen seeds and 7 fine-tune
seeds at std ~0.003, a frozen ESM-2 probe scores 0.55-0.58 while a top-2-layer
fine-tune scores 0.52-0.53. Fine-tuning reliably *loses*, so no threshold
placement makes the task discriminate.

mol-property-adapt's recorded anchors point the other way -- frozen probe 0.6027,
head-only 0.6287, tuned fine-tune 0.6907, a gap of 0.088 AUC. If that survives
multi-seed measurement it is the discriminating task and the better deliverable.
But those anchors are single-run with no error bars, and the file has been edited
during this session (an earlier version recorded gap 0.0227), so the number needs
independent confirmation before anything is built on it.

This applies exactly the protocol that killed the protein task, so the two are
comparable:
    fit on a train slice, early-stop on a held-out val slice, report on the
    private test set. The private set never informs training or selection.

Three arms, matching the three anchor definitions:
    A  frozen backbone + per-task logistic regression   (anchor "base")
    B  frozen backbone + multi-task MLP head            (anchor "head_only")
    C  fine-tune with discriminative LRs                (anchor "reference")

Metric is mean ROC-AUC over the 12 Tox21 assays. Labels are sparse -- 16% of
train and 22% of test entries are NaN -- so every loss and every AUC is computed
only over observed entries. Scoring a NaN as negative would inflate AUC by
inventing true negatives.

Run:  modal run sciml-protein-regression/scripts/modal_mol_headroom.py
"""

from __future__ import annotations

import modal

BASE = "DeepChem/ChemBERTa-77M-MLM"
REVISION = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
N_SEEDS = 5

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0", "transformers==4.49.0", "scikit-learn==1.5.2",
        "pandas==2.2.3", "numpy==1.26.4", "scipy==1.14.1",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_file("spike/split/agent/tox21_train.csv", "/data/tox21_train.csv")
    .add_local_file("spike/split/private/tox21_test.csv", "/data/tox21_test.csv")
    .add_local_file("spike/split/agent/bbbp_train.csv", "/data/bbbp_train.csv")
    .add_local_file("spike/split/private/bbbp_test.csv", "/data/bbbp_test.csv")
)

cache = modal.Volume.from_name("esm-probe-cache", create_if_missing=True)
app = modal.App("mol-headroom")


def _load(ds: str = "tox21"):
    import numpy as np
    import pandas as pd

    tr = pd.read_csv(f"/data/{ds}_train.csv")
    te = pd.read_csv(f"/data/{ds}_test.csv")
    labels = [c for c in tr.columns if c != "smiles"]
    return (
        tr["smiles"].tolist(), tr[labels].to_numpy(np.float64),
        te["smiles"].tolist(), te[labels].to_numpy(np.float64),
        labels,
    )


def _mean_auc(y_true, y_score) -> float:
    """Mean ROC-AUC over tasks, skipping unobserved entries and degenerate tasks."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    aucs = []
    for j in range(y_true.shape[1]):
        obs = ~np.isnan(y_true[:, j])
        yt = y_true[obs, j]
        if obs.sum() < 10 or yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt, y_score[obs, j]))
    return float(np.mean(aucs))


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/cache": cache})
def embed(ds: str) -> str:
    """Frozen ChemBERTa mean-pooled embeddings for both splits, cached."""
    import os

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = f"/cache/mol_emb_{ds}.npz"
    if os.path.exists(path):
        print("cache hit", flush=True)
        return path

    tr_s, _, te_s, _, _ = _load(ds)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = AutoModel.from_pretrained(BASE, revision=REVISION).to("cuda").eval()

    def run(smiles):
        out = []
        with torch.no_grad():
            for i in range(0, len(smiles), 128):
                enc = tok(smiles[i : i + 128], return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to("cuda")
                h = model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1).clamp_min(1e-6)).float().cpu())
        return torch.cat(out).numpy()

    np.savez(path, tr=run(tr_s), te=run(te_s))
    cache.commit()
    print("embedded", flush=True)
    return path


@app.function(image=image, timeout=1800, cpu=4, volumes={"/cache": cache})
def arm_frozen_logreg(ds: str, seed: int) -> dict:
    """Anchor 'base': frozen backbone + one logistic regression per task."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    z = np.load(f"/cache/mol_emb_{ds}.npz")
    _, ytr, _, yte, _ = _load(ds)
    scores = np.zeros_like(yte)
    for j in range(ytr.shape[1]):
        obs = ~np.isnan(ytr[:, j])
        y = ytr[obs, j]
        if y.min() == y.max():
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(z["tr"][obs], y)
        scores[:, j] = clf.predict_proba(z["te"])[:, 1]
    auc = _mean_auc(yte, scores)
    print(f"frozen_logreg seed {seed} -> {auc:.4f}", flush=True)
    return {"ds": ds, "arm": "frozen_logreg", "seed": seed, "private_auc": auc}


@app.function(gpu="A10G", image=image, timeout=1800, volumes={"/cache": cache})
def arm_frozen_head(ds: str, seed: int) -> dict:
    """Anchor 'head_only': frozen backbone + multi-task MLP head."""
    import numpy as np
    import torch

    z = np.load(f"/cache/mol_emb_{ds}.npz")
    _, ytr, _, yte, _ = _load(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    cut = int(0.8 * len(idx))
    fit_i, val_i = idx[:cut], idx[cut:]

    X = torch.tensor(z["tr"], dtype=torch.float32, device="cuda")
    Y = torch.tensor(np.nan_to_num(ytr), dtype=torch.float32, device="cuda")
    M = torch.tensor(~np.isnan(ytr), dtype=torch.float32, device="cuda")
    Xte = torch.tensor(z["te"], dtype=torch.float32, device="cuda")

    torch.manual_seed(seed)
    hid = X.shape[1]
    head = torch.nn.Sequential(
        torch.nn.Linear(hid, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(256, ytr.shape[1]),
    ).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)

    best_val, best_te, patience = -1.0, None, 0
    for _ in range(300):
        head.train()
        opt.zero_grad()
        logit = head(X[fit_i])
        # Masked BCE: unobserved (NaN) entries contribute no gradient.
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            logit, Y[fit_i], reduction="none") * M[fit_i]).sum() / M[fit_i].sum()
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            va = _mean_auc(ytr[val_i], head(X[val_i]).cpu().numpy())
        if va > best_val + 1e-5:
            best_val, patience = va, 0
            with torch.no_grad():
                best_te = head(Xte).cpu().numpy()
        else:
            patience += 1
            if patience >= 20:
                break
    auc = _mean_auc(yte, best_te)
    print(f"frozen_head seed {seed} -> {auc:.4f} (val {best_val:.4f})", flush=True)
    return {"ds": ds, "arm": "frozen_head", "seed": seed, "private_auc": auc, "val_auc": best_val}


@app.function(gpu="A10G", image=image, timeout=5400, volumes={"/cache": cache})
def arm_finetune(ds: str, seed: int) -> dict:
    """Anchor 'reference': fine-tune with discriminative LRs, early stop on val."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tr_s, ytr, te_s, yte, _ = _load(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    cut = int(0.8 * len(idx))
    fit_i, val_i = idx[:cut], idx[cut:]

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    enc_model = AutoModel.from_pretrained(BASE, revision=REVISION).cuda()
    head = torch.nn.Linear(enc_model.config.hidden_size, ytr.shape[1]).cuda()

    # Discriminative learning rates: the pretrained encoder moves an order of
    # magnitude slower than the randomly-initialised head.
    opt = torch.optim.AdamW(
        [{"params": enc_model.parameters(), "lr": 3e-5},
         {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01
    )

    def forward(smiles, train: bool):
        enc = tok(smiles, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to("cuda")
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            h = enc_model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
            return head(pooled)

    def evaluate(smiles, y):
        enc_model.eval(); head.eval()
        outs = []
        for i in range(0, len(smiles), 128):
            outs.append(forward(smiles[i : i + 128], False).float().cpu())
        return _mean_auc(y, torch.cat(outs).numpy())

    Yt = torch.tensor(np.nan_to_num(ytr), dtype=torch.float32, device="cuda")
    Mt = torch.tensor(~np.isnan(ytr), dtype=torch.float32, device="cuda")

    best_val, best_te, patience = -1.0, -1.0, 0
    for epoch in range(1, 21):
        enc_model.train(); head.train()
        order = rng.permutation(fit_i)
        for i in range(0, len(order), 32):
            b = order[i : i + 32]
            opt.zero_grad()
            logit = forward([tr_s[k] for k in b], True)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logit, Yt[b], reduction="none") * Mt[b]).sum() / Mt[b].sum()
            loss.backward()
            opt.step()
        va = evaluate([tr_s[k] for k in val_i], ytr[val_i])
        if va > best_val:
            best_val, patience = va, 0
            best_te = evaluate(te_s, yte)
        else:
            patience += 1
            if patience >= 5:
                break
        print(f"ft seed {seed} epoch {epoch} val={va:.4f}", flush=True)
    print(f"finetune seed {seed} -> {best_te:.4f}", flush=True)
    return {"ds": ds, "arm": "finetune", "seed": seed, "private_auc": best_te, "val_auc": best_val}


@app.local_entrypoint()
def main(dataset: str = "tox21"):
    import json
    import statistics as st
    from pathlib import Path

    embed.remote(dataset)
    seeds = [(dataset, s) for s in range(N_SEEDS)]
    rows = (
        list(arm_frozen_logreg.starmap(seeds))
        + list(arm_frozen_head.starmap(seeds))
        + list(arm_finetune.starmap(seeds))
    )

    def stats(arm):
        v = [r["private_auc"] for r in rows if r["arm"] == arm]
        return {"n": len(v), "mean": round(st.mean(v), 4),
                "std": round(st.stdev(v), 4) if len(v) > 1 else 0.0,
                "min": round(min(v), 4), "max": round(max(v), 4)}

    fz, hd, ft = stats("frozen_logreg"), stats("frozen_head"), stats("finetune")
    best_frozen = max(fz["mean"], hd["mean"])
    pooled = (max(fz["std"], hd["std"]) ** 2 + ft["std"] ** 2) ** 0.5
    gap = ft["mean"] - best_frozen
    out = {
        "frozen_logreg": fz, "frozen_head": hd, "finetune": ft,
        "headroom_vs_best_frozen": round(gap, 4),
        "headroom_sigma": round(gap / pooled, 2) if pooled else None,
        "recorded_anchors": {"base": 0.6027, "head_only": 0.6287, "reference": 0.6907},
        "protein_task_for_contrast": {"frozen": 0.5494, "finetune": 0.5214,
                                      "headroom": -0.028},
        "raw": rows,
    }
    dest = Path(__file__).resolve().parent / f"mol_headroom_{dataset}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print("\n" + "=" * 68)
    print(f"frozen + logreg : {fz}")
    print(f"frozen + head   : {hd}")
    print(f"fine-tuned      : {ft}")
    print(f"\nheadroom vs best frozen: {out['headroom_vs_best_frozen']} "
          f"({out['headroom_sigma']} sigma)")
    print(f"protein task, for contrast: -0.028 (-7.5 sigma)")
    print(f"\nwrote {dest}")
    if gap > 0 and (out["headroom_sigma"] or 0) >= 3:
        print("\n=> fine-tuning wins decisively: this task discriminates.")
    elif gap > 0:
        print("\n=> fine-tuning wins but within noise: needs more seeds or a wider gap.")
    else:
        print("\n=> inverted like the protein task: frozen features already win.")
