"""Is the protein task inverted, or only inverted for the naive recipe?

The question
------------
This task was shelved as inverted: a frozen head beats a fine-tune, so no
threshold placement discriminates. That verdict rests on modal_variance.py, which
measured

    frozen trained head   0.5460 +/- 0.0054  (n=8)
    fine-tune             0.5169 +/- 0.0053  (n=4)

and on modal_cluster_split.py, which reproduced the same sign on a second split at
-7.5 and -12.2 sigma. Both fine-tune arms unfreeze the top two encoder layers and
start from a randomly-initialised head.

The GPU oracle run (jobs/protein-oracle-gpu) does something else and scored
**0.5733** -- above the frozen ceiling, not below it. solution/solve.sh fits a head
on frozen features first, then warm-starts a *full* fine-tune from it. That is
LP-FT, and it is precisely the remedy Kumar et al. 2022 (arXiv:2202.10054) propose
for the feature distortion that produces this exact inversion: fine-tuning
destroys good pretrained features when the head is still random, because early
gradients are dominated by head error.

If that holds up, "inverted" is a property of the recipe rather than the task, and
the shelving decision was measured against the wrong arm.

What this measures
------------------
Both arms, same seeds, same protocol, in one run so the two distributions are
directly comparable rather than assembled from separate sessions:

    frozen   head on frozen CLS features, trained to convergence  (the base)
    lpft     that same head, loaded into the full model, then all parameters
             fine-tuned -- solution/solve.sh's recipe                (the reference)

The lpft arm replicates solve.sh: smooth-L1 on standardised targets, discriminative
LR (head at 3x), 6% warmup then cosine decay to 8%, gradient clipping at 1.0,
best-val checkpoint. Only the data plumbing differs, because features are read from
the Volume rather than recomputed.

Protocol, identical on both arms and identical to modal_variance.py so the numbers
join up with the existing measurements:
    fit    on train.csv.gz split == "train"  (17,922)
    select on train.csv.gz split == "val"    (1,991)
    report on private test.csv.gz            (3,427)
The private set never influences training or selection.

A positive, well-separated band would mean the task is repairable with base = the
frozen ceiling and reference = LP-FT, replacing the tiers with the same continuous
recovery the mol task uses. A negative or noisy one confirms the shelving.

Run:  modal run tasks/sciml-protein-regression/scripts/modal_lpft.py
"""

from __future__ import annotations

import modal

ROOT = "/data"
TASK = "tasks/sciml-protein-regression"
BASE = "facebook/esm2_t6_8M_UR50D"
REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"

N_SEEDS = 5
EPOCHS = 4
BATCH = 8
LR = 7e-5
MAX_LEN = 512

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0", "transformers==4.49.0", "scikit-learn==1.5.2",
        "pandas==2.2.3", "numpy==1.26.4", "scipy==1.14.1",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_file(f"{TASK}/environment/data/train.csv.gz", f"{ROOT}/train.csv.gz")
    .add_local_file(f"{TASK}/tests/private_test/test.csv.gz", f"{ROOT}/test.csv.gz")
)

cache = modal.Volume.from_name("esm-probe-cache", create_if_missing=True)
app = modal.App("esm-lpft")


def _load_splits():
    import gzip

    import pandas as pd

    with gzip.open(f"{ROOT}/train.csv.gz", "rt") as fh:
        train = pd.read_csv(fh)
    with gzip.open(f"{ROOT}/test.csv.gz", "rt") as fh:
        test = pd.read_csv(fh)
    return (train[train["split"] == "train"].reset_index(drop=True),
            train[train["split"] == "val"].reset_index(drop=True),
            test)


def _head(dim, device):
    """EsmClassificationHead's shape, so the fitted head loads straight into it."""
    import torch
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(0.0)
            self.out_proj = nn.Linear(dim, 1)

        def forward(self, x):
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            return self.out_proj(self.dropout(x)).squeeze(-1)

    return Head().to(device)


def _fit_head(xtr, ytr, xva, yva, seed):
    """Train a head on frozen features to convergence. This is the `frozen` arm,
    and also the warm start LP-FT depends on."""
    import numpy as np
    import torch
    import torch.nn as nn
    from scipy.stats import spearmanr

    torch.manual_seed(seed)
    head = _head(xtr.shape[1], "cuda")
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    xt = torch.tensor(xtr, dtype=torch.float32, device="cuda")
    yt = torch.tensor(ytr, dtype=torch.float32, device="cuda")
    xv = torch.tensor(xva, dtype=torch.float32, device="cuda")

    g = torch.Generator(device="cpu").manual_seed(seed)
    best_rho, best_state, best_ep = -1.0, None, 0
    for epoch in range(300):
        head.train()
        order = torch.randperm(len(xt), generator=g).to("cuda")
        for i in range(0, len(order), 256):
            ii = order[i : i + 256]
            loss = nn.functional.smooth_l1_loss(head(xt[ii]), yt[ii], beta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            rho = float(spearmanr(yva, head(xv).cpu().numpy()).statistic)
        if rho > best_rho:
            best_rho, best_ep = rho, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if epoch - best_ep > 20:
            break
    head.load_state_dict(best_state)
    return head, best_rho, best_ep


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/cache": cache})
def frozen_arm(seed: int) -> dict:
    """base: encoder untouched, head trained on frozen CLS features."""
    import numpy as np
    import torch
    from scipy.stats import spearmanr

    z = np.load("/cache/emb_final.npz")
    tr, va, test = _load_splits()
    ytr = tr["target"].to_numpy(np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std())
    ytr_n = ((ytr - mu) / sd).astype(np.float32)
    yva_n = ((va["target"].to_numpy(np.float64) - mu) / sd).astype(np.float32)

    head, best_val, ep = _fit_head(z["tr_cls"], ytr_n, z["va_cls"], yva_n, seed)
    with torch.no_grad():
        pred = head(torch.tensor(z["te_cls"], dtype=torch.float32, device="cuda")).cpu().numpy()
    rho = float(spearmanr(test["target"].to_numpy(np.float64), pred).statistic)
    print(f"[frozen] seed {seed} -> private {rho:.4f} (val {best_val:.4f} @ ep {ep})",
          flush=True)
    return {"arm": "frozen", "seed": seed, "private": rho, "val": best_val}


@app.function(gpu="A10G", image=image, timeout=7200, volumes={"/cache": cache})
def lpft_arm(seed: int) -> dict:
    """reference: LP-FT -- the frozen head above, then a full fine-tune from it."""
    import math

    import numpy as np
    import torch
    import torch.nn as nn
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer, EsmForSequenceClassification

    z = np.load("/cache/emb_final.npz")
    tr, va, test = _load_splits()
    ytr = tr["target"].to_numpy(np.float64)
    yva = va["target"].to_numpy(np.float64)
    yte = test["target"].to_numpy(np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std())
    ytr_n = ((ytr - mu) / sd).astype(np.float32)
    yva_n = ((yva - mu) / sd).astype(np.float32)

    # Stage 1 -- linear probe. Identical to the frozen arm, same seed, so the two
    # arms share a starting point and the delta isolates the fine-tune.
    head, head_val, _ = _fit_head(z["tr_cls"], ytr_n, z["va_cls"], yva_n, seed)
    print(f"[lpft] seed {seed} head val {head_val:.4f}", flush=True)

    # Stage 2 -- full fine-tune, warm-started from that head.
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = EsmForSequenceClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=1, problem_type="regression").cuda()
    model.classifier.load_state_dict(head.state_dict())
    del head

    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    headp = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": LR, "weight_decay": 0.01},
                             {"params": headp, "lr": LR * 3.0, "weight_decay": 0.01}])

    seqs = tr["sequence"].astype(str).tolist()
    steps_per_epoch = math.ceil(len(seqs) / BATCH)
    total = steps_per_epoch * EPOCHS
    warmup = max(1, int(total * 0.06))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.08, 0.5 * (1.0 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    def run_eval(df, y, bs=64):
        model.eval()
        preds = []
        s = df["sequence"].astype(str).tolist()
        with torch.no_grad():
            for i in range(0, len(s), bs):
                enc = tok(s[i : i + bs], return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LEN).to("cuda")
                preds.append(model(**enc).logits.squeeze(-1).float().cpu())
        return float(spearmanr(y, torch.cat(preds).numpy()).statistic)

    g = np.random.default_rng(seed)
    best_val, best_te = -1.0, -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = g.permutation(len(seqs))
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            enc = tok([seqs[k] for k in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to("cuda")
            t = torch.tensor(ytr_n[idx], dtype=torch.float32, device="cuda")
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(
                model(**enc).logits.squeeze(-1), t, beta=0.5)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        rv = run_eval(va, yva)
        print(f"[lpft] seed {seed} epoch {epoch}/{EPOCHS} val={rv:.4f}", flush=True)
        if rv > best_val:
            best_val, best_te = rv, run_eval(test, yte)

    print(f"[lpft] seed {seed} -> private {best_te:.4f}", flush=True)
    return {"arm": "lpft", "seed": seed, "private": best_te, "val": best_val,
            "head_val": head_val}


@app.local_entrypoint()
def main():
    import json
    import statistics as st
    from pathlib import Path

    rows = list(frozen_arm.map(range(N_SEEDS))) + list(lpft_arm.map(range(N_SEEDS)))

    def stat(arm):
        v = [r["private"] for r in rows if r["arm"] == arm]
        return {"n": len(v), "mean": round(st.mean(v), 4),
                "std": round(st.stdev(v), 4) if len(v) > 1 else 0.0,
                "min": round(min(v), 4), "max": round(max(v), 4)}

    fz, ft = stat("frozen"), stat("lpft")
    band = ft["mean"] - fz["mean"]
    pooled = (fz["std"] ** 2 + ft["std"] ** 2) ** 0.5
    sigma = round(band / pooled, 2) if pooled else None

    out = {
        "question": "is the inversion a property of the task, or only of the naive recipe?",
        "frozen": fz, "lpft": ft,
        "band": round(band, 4), "band_sigma": sigma,
        "noise_as_pct_of_band": {
            "frozen": round(fz["std"] / band * 100, 1) if band > 0 else None,
            "lpft": round(ft["std"] / band * 100, 1) if band > 0 else None,
        },
        "for_contrast": {
            "naive_finetune_top2_layers": {"mean": 0.5169, "std": 0.0053, "n": 4,
                                           "source": "modal_variance.py"},
            "frozen_head_prior": {"mean": 0.546, "std": 0.0054, "n": 8,
                                  "source": "modal_variance.py"},
            "gpu_oracle_single_run": 0.5733,
            "graded_frozen_artifact": 0.5358,
            "current_tiers": {"t_weak": 0.3887, "t_strong": 0.45},
        },
        "raw": rows,
    }
    dest = Path(__file__).resolve().parent / "lpft.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print("\n" + "=" * 70)
    print(f"frozen (base)      : {fz}")
    print(f"LP-FT  (reference) : {ft}")
    print(f"band {band:+.4f}  ({sigma} sigma)")
    print(f"\nnaive top-2-layer fine-tune, for contrast: 0.5169 +/- 0.0053 (n=4)")
    print(f"wrote {dest}")
    if band <= 0:
        print("\n=> still inverted even with LP-FT. The shelving stands.")
    elif (sigma or 0) < 3:
        print("\n=> LP-FT is ahead but under 3 sigma: not enough to rebuild a reward on.")
    else:
        print("\n=> NOT a property of the task. The inversion is the naive recipe's.")
        print("   base = frozen ceiling, reference = LP-FT, continuous recovery in")
        print("   place of the tiers -- the mol task's scheme, which discriminates.")
