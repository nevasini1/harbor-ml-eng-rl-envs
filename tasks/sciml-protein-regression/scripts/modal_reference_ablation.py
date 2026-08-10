"""Why is the `reference` anchor 0.042 below a plainer fine-tune?

Why
---
`anchors_private.json` defines reference = "tuned fine-tune" and records 0.6907,
measured by lock_lowdata_anchors.py with the same recipe train_reference.py ships.
modal_mol_headroom.py's fine-tune arm scores 0.7324 +/- 0.0059 (n=5) on identical
data -- frozen_logreg matches to four decimals across both scripts, so the split
and the 2,000-molecule subsample are the same. The 0.042 is entirely recipe.

That matters because reference is the top of the reward scale. At 0.6907 every
competent fine-tune lands past it and clips to 1.0, so the upper half of the band
measures nothing. Raising the number alone is not the fix: reference is *defined*
as what train_reference.py produces, so the shipped solution has to become the
better recipe before the anchor is re-measured from it.

Three factors differ. All three are plausible and one is already measured
elsewhere in this repo:

  pool      CLS token (AutoModelForSequenceClassification's dense+tanh+out_proj)
            vs mean-pooling over tokens. probe_ceiling.json measured mean-pool
            above CLS at every ESM layer (0.4586 vs 0.4217 final); base is
            *already* mean-pooled, so a CLS reference is a different
            representation from the anchor it is scored against.
  body_lr   5e-5 vs 3e-5. Kumar et al. 2022 (arXiv:2202.10054): a faster encoder
            LR distorts pretrained features, which is the mechanism that inverted
            the protein track. This task sits near that boundary.
  schedule  OneCycleLR (warms up to max_lr over the first 10% of steps) vs a
            constant LR. Warmup pushes the encoder hardest while the head is
            still random -- exactly the phase LP-FT argues does the damage.

Design
------
One-factor-at-a-time from the reference recipe, plus the far endpoint:

    reference   cls      5e-5  onecycle    expect ~0.6907
    meanpool    meanpool 5e-5  onecycle
    lr3e5       cls      3e-5  onecycle
    constlr     cls      5e-5  constant
    headroom    meanpool 3e-5  constant    expect ~0.7324

Each single-factor delta attributes part of the 0.042; if they sum to roughly the
full gap the effects are additive, and if they do not there is an interaction the
endpoint cell will expose.

Everything else is held fixed so the deltas are attributable: 20% val split, all
20 epochs run (no early stop, so patience cannot truncate a cell differently),
best-val epoch selected and the private test evaluated at that epoch only, batch
32, max_length 256, head lr 1e-3, weight_decay 0.01, AdamW. The private set never
informs training or selection.

Run:  modal run tasks/sciml-protein-regression/scripts/modal_reference_ablation.py
"""

from __future__ import annotations

import modal

BASE = "DeepChem/ChemBERTa-77M-MLM"
REVISION = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
N_SEEDS = 5
EPOCHS = 20
BS = 32

CELLS = [
    {"name": "reference", "pool": "cls", "body_lr": 5e-5, "sched": "onecycle"},
    {"name": "meanpool", "pool": "meanpool", "body_lr": 5e-5, "sched": "onecycle"},
    {"name": "lr3e5", "pool": "cls", "body_lr": 3e-5, "sched": "onecycle"},
    {"name": "constlr", "pool": "cls", "body_lr": 5e-5, "sched": "constant"},
    {"name": "headroom", "pool": "meanpool", "body_lr": 3e-5, "sched": "constant"},
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0", "transformers==4.49.0", "scikit-learn==1.5.2",
        "pandas==2.2.3", "numpy==1.26.4", "scipy==1.14.1",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_file("research/split/agent/tox21_train.csv", "/data/tox21_train.csv")
    .add_local_file("research/split/private/tox21_test.csv", "/data/tox21_test.csv")
)

cache = modal.Volume.from_name("esm-probe-cache", create_if_missing=True)
app = modal.App("mol-reference-ablation")


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


@app.function(gpu="A10G", image=image, timeout=5400, volumes={"/cache": cache})
def run_cell(cell: dict, seed: int, ds: str = "tox21") -> dict:
    import numpy as np
    import torch
    from transformers import (AutoModel, AutoModelForSequenceClassification,
                              AutoTokenizer)

    tr_s, ytr, te_s, yte, labels = _load(ds)
    n_lab = ytr.shape[1]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    fit_i, val_i = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)) :]

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)

    if cell["pool"] == "cls":
        # HF's RobertaClassificationHead: takes hidden_state[:, 0] (the <s>
        # token), then dense -> tanh -> out_proj. This is what train_reference.py
        # ships and what produced the 0.6907 anchor.
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE, revision=REVISION, num_labels=n_lab,
            problem_type="multi_label_classification").cuda()
        head_p = [p for n, p in model.named_parameters() if n.startswith("classifier")]
        body_p = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
        modules = [model]

        def forward(smiles, train):
            enc = tok(smiles, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda")
            with torch.enable_grad() if train else torch.no_grad():
                return model(**enc).logits
    else:
        enc_model = AutoModel.from_pretrained(BASE, revision=REVISION).cuda()
        head = torch.nn.Linear(enc_model.config.hidden_size, n_lab).cuda()
        head_p, body_p = list(head.parameters()), list(enc_model.parameters())
        modules = [enc_model, head]

        def forward(smiles, train):
            enc = tok(smiles, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda")
            with torch.enable_grad() if train else torch.no_grad():
                h = enc_model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                return head((h * m).sum(1) / m.sum(1).clamp_min(1e-6))

    opt = torch.optim.AdamW(
        [{"params": body_p, "lr": cell["body_lr"]},
         {"params": head_p, "lr": 1e-3}], weight_decay=0.01,
    )
    steps_per_epoch = max(1, -(-len(fit_i) // BS))
    total_steps = EPOCHS * steps_per_epoch
    sched = None
    if cell["sched"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[cell["body_lr"], 1e-3], total_steps=total_steps,
            pct_start=0.1, anneal_strategy="linear")

    Yt = torch.tensor(np.nan_to_num(ytr), dtype=torch.float32, device="cuda")
    Mt = torch.tensor(~np.isnan(ytr), dtype=torch.float32, device="cuda")

    def evaluate(smiles, y):
        for m in modules:
            m.eval()
        outs = []
        for i in range(0, len(smiles), 128):
            outs.append(forward(smiles[i : i + 128], False).float().cpu())
        return _mean_auc(y, torch.cat(outs).numpy())

    step, best_val, best_te, best_ep = 0, -1.0, -1.0, -1
    for epoch in range(1, EPOCHS + 1):
        for m in modules:
            m.train()
        order = rng.permutation(fit_i)
        for i in range(0, len(order), BS):
            b = order[i : i + BS]
            opt.zero_grad()
            logit = forward([tr_s[k] for k in b], True)
            # Masked BCE: unobserved (NaN) entries contribute no gradient.
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logit, Yt[b], reduction="none") * Mt[b]).sum() / Mt[b].sum()
            loss.backward()
            opt.step()
            step += 1
            if sched is not None and step < total_steps:
                sched.step()

        va = evaluate([tr_s[k] for k in val_i], ytr[val_i])
        # Every cell runs all 20 epochs; selection is best-val, so no cell can be
        # truncated at a different point than another.
        if va > best_val:
            best_val, best_ep = va, epoch
            best_te = evaluate(te_s, yte)
        print(f"[{cell['name']}] seed {seed} ep {epoch}/{EPOCHS} val={va:.4f}",
              flush=True)

    print(f"[{cell['name']}] seed {seed} -> {best_te:.4f} "
          f"(val {best_val:.4f} @ ep {best_ep})", flush=True)
    return {"cell": cell["name"], "seed": seed, "private_auc": best_te,
            "val_auc": best_val, "best_epoch": best_ep, **{
                k: cell[k] for k in ("pool", "body_lr", "sched")}}


@app.local_entrypoint()
def main(dataset: str = "tox21"):
    import json
    import statistics as st
    from pathlib import Path

    jobs = [(c, s, dataset) for c in CELLS for s in range(N_SEEDS)]
    rows = list(run_cell.starmap(jobs))

    def stats(name):
        v = [r["private_auc"] for r in rows if r["cell"] == name]
        return {"n": len(v), "mean": round(st.mean(v), 4),
                "std": round(st.stdev(v), 4) if len(v) > 1 else 0.0,
                "min": round(min(v), 4), "max": round(max(v), 4)}

    cells = {c["name"]: stats(c["name"]) for c in CELLS}
    ref = cells["reference"]["mean"]
    deltas = {n: round(s["mean"] - ref, 4) for n, s in cells.items()}
    single = ["meanpool", "lr3e5", "constlr"]
    additive = round(sum(deltas[n] for n in single), 4)

    out = {
        "held_fixed": {
            "val_frac": 0.2, "epochs": EPOCHS, "early_stop": False,
            "selection": "best val epoch; private test evaluated at that epoch",
            "batch": BS, "max_length": 256, "head_lr": 1e-3, "weight_decay": 0.01,
        },
        "cells": {c["name"]: {**cells[c["name"]],
                              **{k: c[k] for k in ("pool", "body_lr", "sched")}}
                  for c in CELLS},
        "delta_vs_reference": deltas,
        "sum_of_single_factor_deltas": additive,
        "endpoint_delta": deltas["headroom"],
        "interaction": round(deltas["headroom"] - additive, 4),
        "targets": {"recorded_reference_anchor": 0.6907,
                    "modal_mol_headroom_finetune": 0.7324,
                    "gap_to_explain": 0.0417},
        "raw": rows,
    }
    dest = Path(__file__).resolve().parent / "reference_ablation.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print("\n" + "=" * 74)
    print(f"{'cell':<11}{'pool':<10}{'body_lr':<9}{'sched':<10}"
          f"{'mean':<9}{'std':<8}{'delta':<8}")
    for c in CELLS:
        s = cells[c["name"]]
        print(f"{c['name']:<11}{c['pool']:<10}{c['body_lr']:<9.0e}{c['sched']:<10}"
              f"{s['mean']:<9.4f}{s['std']:<8.4f}{deltas[c['name']]:+.4f}")
    print("=" * 74)
    print(f"gap to explain          : +0.0417  (0.6907 -> 0.7324)")
    print(f"endpoint cell delta     : {deltas['headroom']:+.4f}")
    print(f"sum of single factors   : {additive:+.4f}")
    print(f"interaction             : {out['interaction']:+.4f}")
    print(f"\nwrote {dest}")

    win = max(cells, key=lambda n: cells[n]["mean"])
    print(f"\n=> best recipe: {win} at {cells[win]['mean']:.4f}. "
          f"Port it into solution/train_reference.py, then re-measure the anchor "
          f"from that script before re-placing the band.")
