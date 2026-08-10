"""Put error bars on the headroom between a frozen probe and a fine-tuned model.

The question
------------
A frozen ESM-2-8M + trained head, produced by a real Codex run, was graded at
0.5358 on the private test set. The best fine-tuned result ever measured on this
task is 0.5709 (job 02-00). That is a headroom of 0.035 Spearman between "do
nothing clever" and "the strongest solution we have ever seen".

If run-to-run noise is comparable to 0.035, the task cannot separate a capable
agent from a lazy one, and no placement of t_weak / t_strong repairs that. This
measures the noise on both sides so the gap can be judged against it.

What it fixes from modal_probe_ceiling.py
-----------------------------------------
That script's heads all stopped at `selected_epoch: 300` -- the hard cap -- with
validation Spearman still improving, so 0.4316 / 0.4973 were undertrained lower
bounds, not ceilings. Here the head trains in mini-batches with early stopping on
patience, so each seed converges before it is recorded.

Protocol (identical on both arms, so the two distributions are comparable)
    fit    on train.csv.gz split == "train"  (17,922)
    select on train.csv.gz split == "val"    (1,991)   <- early stopping only
    report on private test.csv.gz            (3,427)
The private set never influences training or selection.

Run:  modal run tasks/sciml-protein-regression/scripts/modal_variance.py
"""

from __future__ import annotations

import modal

ROOT = "/data"
TASK = "tasks/sciml-protein-regression"
BASE = "facebook/esm2_t6_8M_UR50D"
REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"

N_FROZEN_SEEDS = 8
N_FINETUNE_SEEDS = 4

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "scikit-learn==1.5.2",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scipy==1.14.1",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_file(f"{TASK}/environment/data/train.csv.gz", f"{ROOT}/train.csv.gz")
    .add_local_file(f"{TASK}/tests/private_test/test.csv.gz", f"{ROOT}/test.csv.gz")
)

cache = modal.Volume.from_name("esm-probe-cache", create_if_missing=True)
app = modal.App("esm-variance")


def _load_splits():
    import gzip

    import pandas as pd

    with gzip.open(f"{ROOT}/train.csv.gz", "rt") as fh:
        train = pd.read_csv(fh)
    with gzip.open(f"{ROOT}/test.csv.gz", "rt") as fh:
        test = pd.read_csv(fh)
    tr = train[train["split"] == "train"].reset_index(drop=True)
    va = train[train["split"] == "val"].reset_index(drop=True)
    return tr, va, test


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/cache": cache})
def extract() -> str:
    """Embed all three splits once (final layer, CLS + mean-pool) into the Volume."""
    import os

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = "/cache/emb_final.npz"
    if os.path.exists(path):
        print("cache hit", flush=True)
        return path

    tr, va, test = _load_splits()
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = AutoModel.from_pretrained(BASE, revision=REVISION).to("cuda").eval()

    @torch.no_grad()
    def embed(seqs, bs=64):
        cls_o, mean_o = [], []
        for i in range(0, len(seqs), bs):
            chunk = seqs[i : i + bs]
            enc = tok(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to("cuda")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            m[:, 0] = 0
            for j, s in enumerate(chunk):
                m[j, min(len(s) + 1, m.shape[1] - 1)] = 0
            cls_o.append(h[:, 0].float().cpu())
            mean_o.append(((h * m).sum(1) / m.sum(1).clamp_min(1e-6)).float().cpu())
        return torch.cat(cls_o).numpy(), torch.cat(mean_o).numpy()

    out = {}
    for name, df in (("tr", tr), ("va", va), ("te", test)):
        c, m = embed(df["sequence"].astype(str).tolist())
        out[f"{name}_cls"], out[f"{name}_mean"] = c, m
        print(f"embedded {name}: {c.shape}", flush=True)

    np.savez(path, **out)
    cache.commit()
    return path


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/cache": cache})
def frozen_seed(seed: int) -> dict:
    """Fit a head on frozen features to convergence. One seed."""
    import numpy as np
    import torch
    from scipy.stats import spearmanr

    z = np.load("/cache/emb_final.npz")
    tr, va, test = _load_splits()
    ytr = tr["target"].to_numpy(np.float64)
    yva = va["target"].to_numpy(np.float64)
    yte = test["target"].to_numpy(np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std())

    best_overall = {"private": -1.0}
    for pooling in ("cls", "mean"):
        Xtr = torch.tensor(z[f"tr_{pooling}"], dtype=torch.float32, device="cuda")
        Xva = torch.tensor(z[f"va_{pooling}"], dtype=torch.float32, device="cuda")
        Xte = torch.tensor(z[f"te_{pooling}"], dtype=torch.float32, device="cuda")
        ttr = torch.tensor((ytr - mu) / sd, dtype=torch.float32, device="cuda")

        torch.manual_seed(seed)
        hid = Xtr.shape[1]
        head = torch.nn.Sequential(
            torch.nn.Linear(hid, hid), torch.nn.Tanh(), torch.nn.Linear(hid, 1)
        ).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)

        # Mini-batch with early stopping: the previous script capped at 300 full-batch
        # steps and was still improving, which understated every frozen number.
        g = torch.Generator(device="cpu").manual_seed(seed)
        best_val, best_te, patience = -1.0, None, 0
        for epoch in range(300):
            head.train()
            perm = torch.randperm(Xtr.shape[0], generator=g).cuda()
            for i in range(0, len(perm), 256):
                idx = perm[i : i + 256]
                opt.zero_grad()
                torch.nn.functional.mse_loss(head(Xtr[idx]).squeeze(-1), ttr[idx]).backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                rv = float(spearmanr(head(Xva).squeeze(-1).cpu().numpy(), yva).statistic)
            if rv > best_val + 1e-5:
                best_val, patience = rv, 0
                with torch.no_grad():
                    best_te = head(Xte).squeeze(-1).cpu().numpy()
            else:
                patience += 1
                if patience >= 15:
                    break
        r = float(spearmanr(best_te, yte).statistic)
        if r > best_overall["private"]:
            best_overall = {"private": r, "val": best_val, "pooling": pooling,
                            "epochs": epoch + 1}

    print(f"seed {seed} frozen -> {best_overall}", flush=True)
    return {"seed": seed, "arm": "frozen", **best_overall}


@app.function(gpu="A10G", image=image, timeout=5400, volumes={"/cache": cache})
def finetune_seed(seed: int) -> dict:
    """Fine-tune top-2 encoder layers + head, mirroring the recipe Codex used."""
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer, EsmForSequenceClassification

    tr, va, test = _load_splits()
    ytr = tr["target"].to_numpy(np.float64)
    yva = va["target"].to_numpy(np.float64)
    yte = test["target"].to_numpy(np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std())

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = EsmForSequenceClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=1, problem_type="regression"
    ).cuda()

    n_layers = model.config.num_hidden_layers
    for name, p in model.named_parameters():
        keep = "classifier" in name or any(
            f"layer.{i}." in name for i in range(n_layers - 2, n_layers)
        )
        p.requires_grad = keep

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-5, weight_decay=0.01
    )

    def run_eval(df, y, crop=512, bs=64):
        model.eval()
        preds = []
        seqs = df["sequence"].astype(str).tolist()
        with torch.no_grad():
            for i in range(0, len(seqs), bs):
                enc = tok(
                    seqs[i : i + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=crop,
                ).to("cuda")
                preds.append(model(**enc).logits.squeeze(-1).float().cpu())
        return float(spearmanr(torch.cat(preds).numpy(), y).statistic)

    seqs = tr["sequence"].astype(str).tolist()
    tgt = (ytr - mu) / sd
    g = np.random.default_rng(seed)
    best_val, best_te = -1.0, -1.0
    for epoch, crop in enumerate((256, 384, 512), start=1):
        model.train()
        order = g.permutation(len(seqs))
        for i in range(0, len(order), 16):
            idx = order[i : i + 16]
            enc = tok(
                [seqs[k] for k in idx], return_tensors="pt", padding=True,
                truncation=True, max_length=crop,
            ).to("cuda")
            t = torch.tensor(tgt[idx], dtype=torch.float32, device="cuda")
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(
                model(**enc).logits.squeeze(-1), t
            )
            loss.backward()
            opt.step()
        rv = run_eval(va, yva)
        print(f"seed {seed} epoch {epoch} (crop {crop}) val={rv:.4f}", flush=True)
        if rv > best_val:
            best_val, best_te = rv, run_eval(test, yte)

    print(f"seed {seed} finetune -> private {best_te:.4f}", flush=True)
    return {"seed": seed, "arm": "finetune", "private": best_te, "val": best_val}


@app.local_entrypoint()
def main():
    import json
    import statistics as st
    from pathlib import Path

    print("extracting frozen features (cached in Volume after first run)...")
    extract.remote()

    print(f"\nfrozen probe x{N_FROZEN_SEEDS} seeds...")
    frozen = list(frozen_seed.map(range(N_FROZEN_SEEDS)))
    print(f"\nfine-tune x{N_FINETUNE_SEEDS} seeds...")
    finetuned = list(finetune_seed.map(range(N_FINETUNE_SEEDS)))

    fz = [r["private"] for r in frozen]
    ft = [r["private"] for r in finetuned]

    def stats(v):
        return {
            "n": len(v),
            "mean": round(st.mean(v), 4),
            "std": round(st.stdev(v), 4) if len(v) > 1 else 0.0,
            "min": round(min(v), 4),
            "max": round(max(v), 4),
        }

    gap = st.mean(ft) - st.mean(fz)
    pooled = (st.stdev(fz) ** 2 + st.stdev(ft) ** 2) ** 0.5
    out = {
        "frozen": stats(fz),
        "finetune": stats(ft),
        "headroom_mean_gap": round(gap, 4),
        "pooled_std": round(pooled, 4),
        "gap_in_pooled_sigma": round(gap / pooled, 2) if pooled else None,
        "graded_reference_frozen": 0.5358,
        "graded_reference_finetune_02_00": 0.5709,
        "t_weak": 0.3887,
        "t_strong": 0.45,
        "raw": frozen + finetuned,
    }
    dest = Path(__file__).resolve().parent / "variance.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print("\n" + "=" * 60)
    print(f"frozen   : {out['frozen']}")
    print(f"finetune : {out['finetune']}")
    print(f"headroom : {out['headroom_mean_gap']}  ({out['gap_in_pooled_sigma']} pooled sigma)")
    print(f"wrote {dest}")
    if out["gap_in_pooled_sigma"] is not None and out["gap_in_pooled_sigma"] < 2:
        print("\n=> gap is within ~2 sigma of noise: the task cannot reliably")
        print("   separate a frozen probe from a fine-tuned solution.")
    else:
        print("\n=> gap is well outside noise: the tiers can discriminate.")
