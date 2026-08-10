"""Measure the real frozen-probe ceiling on the private test set, on a GPU.

Why this exists
---------------
calibrate_tiers.py derives t_weak from ONE frozen baseline: mean-pooled residues
with the CLS token explicitly masked out (`m[:, 0] = 0`), fed to a linear RidgeCV,
fitted on 4,000 of 17,922 training rows. That yields 0.3887, which is written to
tiers.json as `frozen_probe_spearman` and used as t_weak.

Two independent Codex runs instead replicated the real HF regression head --
final-layer CLS -> dense -> tanh -> output -- and reported 0.521 and 0.5491 on
their own validation split. Same "frozen encoder, train a head" idea, materially
stronger probe. If that transfers to the private test set, then t_strong (0.45)
is clearable with no fine-tuning at all and the 1.0 tier stops meaning anything.

Both high numbers are validation-split figures, though, and the private split
looks harder (the 14-08 oracle scored 0.4312 there). So the question this answers
is narrow and specific: what does a frozen probe score on the PRIVATE test set,
measured the way an agent would actually build it?

Protocol
--------
Honest split discipline, so the number is comparable to a graded submission:
    fit    on train.csv.gz split == "train"  (17,922)
    select on train.csv.gz split == "val"    (1,991)
    report on private test.csv.gz            (3,427)
Nothing is selected on the private test set.

Variant A (mean-pool + Ridge) reproduces calibrate_tiers.py and is the CONTROL:
if it does not land near 0.3887, this pipeline diverges from the CPU one and no
other row in the table should be trusted.

Run:  modal run tasks/sciml-protein-regression/scripts/modal_probe_ceiling.py
"""

from __future__ import annotations

import modal

ROOT = "/data"
TASK = "tasks/sciml-protein-regression"

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
    # ESM-2-8M is 8M params; a T4 is ample and cheapest. Pin the same revision
    # the task and grader use so representations are identical.
    .env({"HF_HOME": "/cache/hf"})
    .add_local_file(f"{TASK}/environment/data/train.csv.gz", f"{ROOT}/train.csv.gz")
    .add_local_file(f"{TASK}/tests/private_test/test.csv.gz", f"{ROOT}/test.csv.gz")
)

cache = modal.Volume.from_name("esm-probe-cache", create_if_missing=True)
app = modal.App("esm-probe-ceiling")

BASE = "facebook/esm2_t6_8M_UR50D"
REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"


@app.function(gpu="T4", image=image, timeout=3600, volumes={"/cache": cache})
def measure() -> dict:
    import gzip
    import json

    import numpy as np
    import pandas as pd
    import torch
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda"

    with gzip.open(f"{ROOT}/train.csv.gz", "rt") as fh:
        train = pd.read_csv(fh)
    with gzip.open(f"{ROOT}/test.csv.gz", "rt") as fh:
        test = pd.read_csv(fh)

    tr = train[train["split"] == "train"].reset_index(drop=True)
    va = train[train["split"] == "val"].reset_index(drop=True)
    print(f"fit={len(tr)} select={len(va)} report={len(test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = AutoModel.from_pretrained(BASE, revision=REVISION).to(dev).eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"layers={n_layers} hidden={hidden}", flush=True)

    @torch.no_grad()
    def embed(seqs: list[str], bs: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """Return (cls, meanpool) stacks of shape [n_hidden_states, n_seq, hidden].

        meanpool masks padding, CLS and EOS exactly as calibrate_tiers.py does, so
        variant A is a faithful reproduction rather than an approximation.
        """
        cls_out, mean_out = [], []
        for i in range(0, len(seqs), bs):
            chunk = seqs[i : i + bs]
            enc = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states  # tuple L+1
            m = enc["attention_mask"].unsqueeze(-1).float()
            m[:, 0] = 0  # drop CLS from the mean, matching the CPU script
            for j, s in enumerate(chunk):
                end = min(len(s) + 1, m.shape[1] - 1)  # drop EOS
                m[j, end] = 0
            denom = m.sum(1).clamp_min(1e-6)
            cls_out.append(torch.stack([h[:, 0] for h in hs]).float().cpu())
            mean_out.append(
                torch.stack([(h * m).sum(1) / denom for h in hs]).float().cpu()
            )
            if i % (bs * 40) == 0:
                print(f"  embedded {i}/{len(seqs)}", flush=True)
        return (
            torch.cat(cls_out, dim=1).numpy(),
            torch.cat(mean_out, dim=1).numpy(),
        )

    print("embedding fit split...", flush=True)
    tr_cls, tr_mean = embed(tr["sequence"].astype(str).tolist())
    print("embedding select split...", flush=True)
    va_cls, va_mean = embed(va["sequence"].astype(str).tolist())
    print("embedding private test...", flush=True)
    te_cls, te_mean = embed(test["sequence"].astype(str).tolist())

    ytr = tr["target"].to_numpy(dtype=np.float64)
    yva = va["target"].to_numpy(dtype=np.float64)
    yte = test["target"].to_numpy(dtype=np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std())

    def rho(pred, y) -> float:
        return float(spearmanr(pred, y).statistic)

    def ridge(Etr, Ete) -> float:
        alphas = np.logspace(-3, 6, 20)
        return rho(RidgeCV(alphas=alphas).fit(Etr, ytr).predict(Ete), yte)

    def nonlinear_head(Etr, Eva, Ete, epochs: int = 300) -> tuple[float, int]:
        """EsmClassificationHead topology: dense -> tanh -> out_proj, on frozen feats.

        Selects the epoch by validation Spearman, then reports private test at that
        epoch. The private set never influences selection.
        """
        Xtr = torch.tensor(Etr, dtype=torch.float32, device=dev)
        Xva = torch.tensor(Eva, dtype=torch.float32, device=dev)
        Xte = torch.tensor(Ete, dtype=torch.float32, device=dev)
        ttr = torch.tensor((ytr - mu) / sd, dtype=torch.float32, device=dev)

        torch.manual_seed(0)
        head = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        ).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)

        best = (-1.0, 0, None)
        for ep in range(1, epochs + 1):
            head.train()
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(head(Xtr).squeeze(-1), ttr)
            loss.backward()
            opt.step()
            if ep % 10 == 0:
                head.eval()
                with torch.no_grad():
                    r = rho(head(Xva).squeeze(-1).cpu().numpy(), yva)
                if r > best[0]:
                    with torch.no_grad():
                        best = (r, ep, head(Xte).squeeze(-1).cpu().numpy())
        return rho(best[2], yte), best[1]

    rows = []
    # Final layer is what a deployable HF head reads; sweep all layers too.
    for layer in range(n_layers + 1):
        rows.append(
            {
                "layer": layer,
                "meanpool_ridge": round(ridge(tr_mean[layer], te_mean[layer]), 4),
                "cls_ridge": round(ridge(tr_cls[layer], te_cls[layer]), 4),
            }
        )
        print(f"layer {layer}: {rows[-1]}", flush=True)

    # Nonlinear head only on the layers that matter, to keep runtime tight.
    head_rows = {}
    for name, (A, B, C) in {
        "cls_nonlinear_final": (tr_cls[-1], va_cls[-1], te_cls[-1]),
        "meanpool_nonlinear_final": (tr_mean[-1], va_mean[-1], te_mean[-1]),
    }.items():
        r, ep = nonlinear_head(A, B, C)
        head_rows[name] = {"private_spearman": round(r, 4), "selected_epoch": ep}
        print(f"{name}: {head_rows[name]}", flush=True)

    control = rows[-1]["meanpool_ridge"]
    result = {
        "n_fit": len(tr),
        "n_select": len(va),
        "n_private_test": len(test),
        "per_layer_linear": rows,
        "nonlinear_heads": head_rows,
        "control_meanpool_ridge_final_layer": control,
        "tiers_json_claims_frozen_probe": 0.3887,
        "t_strong_in_tiers_json": 0.45,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    out = measure.remote()
    # Deliberately NOT under tests/: that whole directory is the verifier build
    # context (COPY . /tests/), and calibration output has no business in the
    # grader image.
    dest = Path(__file__).resolve().parent / "probe_ceiling.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")

    best_linear = max(
        max(r["meanpool_ridge"], r["cls_ridge"]) for r in out["per_layer_linear"]
    )
    best_head = max(v["private_spearman"] for v in out["nonlinear_heads"].values())
    print(f"\ncontrol (should be ~0.3887): {out['control_meanpool_ridge_final_layer']}")
    print(f"best frozen linear probe    : {best_linear}")
    print(f"best frozen nonlinear head  : {best_head}")
    print(f"t_strong currently          : {out['t_strong_in_tiers_json']}")
    if best_head >= out["t_strong_in_tiers_json"]:
        print("\n=> a FROZEN probe clears t_strong: the 1.0 tier needs no fine-tuning")
    else:
        print("\n=> frozen probe stays below t_strong: the tier boundaries hold")
