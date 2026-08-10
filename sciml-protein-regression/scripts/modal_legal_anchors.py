"""Re-measure both anchors inside the submission contract the verifier enforces.

Why
---
tests/grade.py loads a submission with AutoModelForSequenceClassification and
reads `.logits`, and instruction.md promises no agent code runs in the verifier.
The base model is model_type roberta, so a submission *is* a
RobertaForSequenceClassification, whose RobertaClassificationHead reads
`features[:, 0, :]` -- the <s>/CLS token. Mean-pooling is not expressible in that
architecture.

Every anchor measured so far is therefore off-contract:

    base       0.6430   MLP on mean-pooled embeddings   -> not a legal submission
    reference  0.7324   AutoModel + mean-pool + Linear  -> not a legal submission

Setting reference to 0.7324 would put reward 1.0 permanently out of reach, and
base at 0.6430 is a ceiling drawn over methods no agent is allowed to submit. The
reference_ablation result that mean-pooling is worth +0.0215 is real but
unshippable: it is a measurement taken outside the rules.

So both anchors are re-measured here using only architectures an agent could
actually submit. Every arm below is a legal RobertaForSequenceClassification.

Arms
----
    legal_frozen_logreg  encoder frozen, logistic regression per task on the CLS
                         embedding. The CLS counterpart of the old 0.6027.
    legal_frozen_head    encoder frozen, RobertaClassificationHead trained on the
                         CLS embedding -- dense -> tanh -> out_proj, replicated
                         exactly so the score corresponds to a real submission.
                         This is the legal trivial ceiling, i.e. the new `base`.
    legal_finetune       full fine-tune, CLS head, body lr 3e-5 + OneCycle. The
                         best of the three legal cells in reference_ablation.json
                         (0.7019 +/- 0.0055, ahead of 0.7006 and 0.6941 but
                         within noise of both). This is the new `reference`.

Freezing the encoder and training only the head is explicitly allowed by the task
rules, which is exactly why it sets the floor: it is the best score reachable
without adapting the encoder, and it must earn zero.

The two frozen arms train on CLS embeddings cached once. That is mathematically
identical to freezing the encoder and running full forward passes, and it lets the
head train to convergence rather than being cut short by GPU budget -- the trivial
ceiling has to be measured at its best or it is not a ceiling.

Protocol matches the earlier runs so the numbers are comparable: fit on 80%,
early-stop or select on the held-out 20%, report on the private test set, which
never informs training or selection.

Run:  modal run sciml-protein-regression/scripts/modal_legal_anchors.py
"""

from __future__ import annotations

import modal

BASE = "DeepChem/ChemBERTa-77M-MLM"
REVISION = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
N_SEEDS = 5
EPOCHS = 20
BS = 32
BODY_LR = 3e-5
HEAD_LR = 1e-3

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0", "transformers==4.49.0", "scikit-learn==1.5.2",
        "pandas==2.2.3", "numpy==1.26.4", "scipy==1.14.1",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_file("spike/split/agent/tox21_train.csv", "/data/tox21_train.csv")
    .add_local_file("spike/split/private/tox21_test.csv", "/data/tox21_test.csv")
)

cache = modal.Volume.from_name("esm-probe-cache", create_if_missing=True)
app = modal.App("mol-legal-anchors")


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
def embed_cls(ds: str = "tox21") -> str:
    """CLS-token embeddings -- the only representation the head is allowed to see."""
    import os

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = f"/cache/mol_cls_emb_{ds}.npz"
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
                out.append(model(**enc).last_hidden_state[:, 0, :].float().cpu())
        return torch.cat(out).numpy()

    np.savez(path, tr=run(tr_s), te=run(te_s))
    cache.commit()
    print("embedded CLS", flush=True)
    return path


@app.function(image=image, timeout=1800, cpu=4, volumes={"/cache": cache})
def arm_legal_frozen_logreg(seed: int, ds: str = "tox21") -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    z = np.load(f"/cache/mol_cls_emb_{ds}.npz")
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
    print(f"legal_frozen_logreg seed {seed} -> {auc:.4f}", flush=True)
    return {"arm": "legal_frozen_logreg", "seed": seed, "private_auc": auc}


@app.function(gpu="A10G", image=image, timeout=1800, volumes={"/cache": cache})
def arm_legal_frozen_head(seed: int, ds: str = "tox21") -> dict:
    """The legal trivial ceiling: encoder frozen, RobertaClassificationHead trained."""
    import numpy as np
    import torch

    z = np.load(f"/cache/mol_cls_emb_{ds}.npz")
    _, ytr, _, yte, _ = _load(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    fit_i, val_i = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)) :]

    X = torch.tensor(z["tr"], dtype=torch.float32, device="cuda")
    Xte = torch.tensor(z["te"], dtype=torch.float32, device="cuda")
    Y = torch.tensor(np.nan_to_num(ytr), dtype=torch.float32, device="cuda")
    M = torch.tensor(~np.isnan(ytr), dtype=torch.float32, device="cuda")

    torch.manual_seed(seed)
    hid = X.shape[1]

    # Exactly RobertaClassificationHead: dropout -> dense -> tanh -> dropout ->
    # out_proj. Anything else would score a model that cannot be submitted.
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(hid, hid)
            self.dropout = torch.nn.Dropout(0.1)
            self.out_proj = torch.nn.Linear(hid, ytr.shape[1])

        def forward(self, x):
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    head = Head().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=1e-2)

    best_val, best_te, patience = -1.0, None, 0
    for _ in range(300):
        head.train()
        opt.zero_grad()
        logit = head(X[fit_i])
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
    print(f"legal_frozen_head seed {seed} -> {auc:.4f} (val {best_val:.4f})", flush=True)
    return {"arm": "legal_frozen_head", "seed": seed, "private_auc": auc,
            "val_auc": best_val}


@app.function(gpu="A10G", image=image, timeout=5400, volumes={"/cache": cache})
def arm_legal_finetune(seed: int, ds: str = "tox21") -> dict:
    """The legal reference: full fine-tune through the CLS classification head."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tr_s, ytr, te_s, yte, _ = _load(ds)
    n_lab = ytr.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    fit_i, val_i = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)) :]

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=n_lab,
        problem_type="multi_label_classification").cuda()

    head_p = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body_p = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW(
        [{"params": body_p, "lr": BODY_LR}, {"params": head_p, "lr": HEAD_LR}],
        weight_decay=0.01)
    steps_per_epoch = max(1, -(-len(fit_i) // BS))
    total_steps = EPOCHS * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[BODY_LR, HEAD_LR], total_steps=total_steps, pct_start=0.1,
        anneal_strategy="linear")

    Yt = torch.tensor(np.nan_to_num(ytr), dtype=torch.float32, device="cuda")
    Mt = torch.tensor(~np.isnan(ytr), dtype=torch.float32, device="cuda")

    def forward(smiles, train):
        enc = tok(smiles, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to("cuda")
        with torch.enable_grad() if train else torch.no_grad():
            return model(**enc).logits

    def evaluate(smiles, y):
        model.eval()
        outs = []
        for i in range(0, len(smiles), 128):
            outs.append(forward(smiles[i : i + 128], False).float().cpu())
        return _mean_auc(y, torch.cat(outs).numpy())

    step, best_val, best_te, best_ep = 0, -1.0, -1.0, -1
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(fit_i)
        for i in range(0, len(order), BS):
            b = order[i : i + BS]
            opt.zero_grad()
            logit = forward([tr_s[k] for k in b], True)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logit, Yt[b], reduction="none") * Mt[b]).sum() / Mt[b].sum()
            loss.backward()
            opt.step()
            step += 1
            if step < total_steps:
                sched.step()
        va = evaluate([tr_s[k] for k in val_i], ytr[val_i])
        if va > best_val:
            best_val, best_ep = va, epoch
            best_te = evaluate(te_s, yte)
        print(f"legal_ft seed {seed} ep {epoch}/{EPOCHS} val={va:.4f}", flush=True)
    print(f"legal_finetune seed {seed} -> {best_te:.4f} (val {best_val:.4f} "
          f"@ ep {best_ep})", flush=True)
    return {"arm": "legal_finetune", "seed": seed, "private_auc": best_te,
            "val_auc": best_val, "best_epoch": best_ep}


@app.local_entrypoint()
def main(dataset: str = "tox21"):
    import json
    import statistics as st
    from pathlib import Path

    embed_cls.remote(dataset)
    seeds = [(s, dataset) for s in range(N_SEEDS)]
    rows = (
        list(arm_legal_frozen_logreg.starmap(seeds))
        + list(arm_legal_frozen_head.starmap(seeds))
        + list(arm_legal_finetune.starmap(seeds))
    )

    def stats(arm):
        v = [r["private_auc"] for r in rows if r["arm"] == arm]
        return {"n": len(v), "mean": round(st.mean(v), 4),
                "std": round(st.stdev(v), 4) if len(v) > 1 else 0.0,
                "min": round(min(v), 4), "max": round(max(v), 4)}

    lg, hd, ft = (stats("legal_frozen_logreg"), stats("legal_frozen_head"),
                  stats("legal_finetune"))
    base = max(lg["mean"], hd["mean"])          # ceiling of the trivial rungs
    base_arm = "legal_frozen_head" if hd["mean"] >= lg["mean"] else "legal_frozen_logreg"
    band = ft["mean"] - base
    base_std = hd["std"] if base_arm == "legal_frozen_head" else lg["std"]
    pooled = (base_std ** 2 + ft["std"] ** 2) ** 0.5

    out = {
        "contract": "RobertaForSequenceClassification + .logits (CLS head); "
                    "every arm here is a legal submission",
        "legal_frozen_logreg": lg,
        "legal_frozen_head": hd,
        "legal_finetune": ft,
        "proposed_anchors": {
            "base_auc": round(base, 4), "base_arm": base_arm,
            "reference_auc": round(ft["mean"], 4),
            "reference_arm": "legal_finetune",
            "band": round(band, 4),
            "band_sigma": round(band / pooled, 2) if pooled else None,
        },
        "noise_as_pct_of_band": {
            "frozen_head": round(hd["std"] / band * 100, 1) if band > 0 else None,
            "finetune": round(ft["std"] / band * 100, 1) if band > 0 else None,
        },
        "superseded_off_contract": {
            "base_meanpool_mlp": 0.6430, "reference_meanpool_linear": 0.7324,
            "why": "mean-pooling is not expressible in RobertaForSequenceClassification",
        },
        "raw": rows,
    }
    dest = Path(__file__).resolve().parent / "legal_anchors.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print("\n" + "=" * 70)
    print(f"legal_frozen_logreg : {lg}")
    print(f"legal_frozen_head   : {hd}")
    print(f"legal_finetune      : {ft}")
    print("=" * 70)
    print(f"base      = {base:.4f}  ({base_arm})")
    print(f"reference = {ft['mean']:.4f}  (legal_finetune)")
    print(f"band      = {band:+.4f}   ({out['proposed_anchors']['band_sigma']} sigma)")
    print(f"noise     : frozen_head {out['noise_as_pct_of_band']['frozen_head']}% "
          f"of band, finetune {out['noise_as_pct_of_band']['finetune']}%")
    print(f"\nwrote {dest}")
    if band <= 0:
        print("\n=> INVERTED under the contract: the task does not discriminate.")
    elif (out["proposed_anchors"]["band_sigma"] or 0) < 3:
        print("\n=> ordered but tight: gap is under 3 sigma.")
    else:
        print("\n=> ordered and separated under the contract.")
