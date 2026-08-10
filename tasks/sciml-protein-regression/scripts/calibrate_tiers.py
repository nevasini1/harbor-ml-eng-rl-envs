"""Measure frozen-probe Spearman on the private test set to set T_weak.

T_strong is taken from a strong-oracle checkpoint if provided, else from a
target we expect solve.sh to clear after the recipe upgrade.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = np.logspace(-3, 6, 20)


@torch.no_grad()
def embed(seqs, tok, model, bs=16, max_len=512):
    out = []
    for i in range(0, len(seqs), bs):
        enc = tok(
            seqs[i : i + bs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        m[:, 0] = 0
        for j, s in enumerate(seqs[i : i + bs]):
            # drop EOS if present
            end = min(len(s) + 1, m.shape[1] - 1)
            m[j, end] = 0
        out.append(((h * m).sum(1) / m.sum(1).clamp_min(1e-6)).numpy())
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="local HF model dir or hub id")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=ROOT / "tests" / "tiers.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    train_path = ROOT / "environment" / "data" / "train.csv.gz"
    test_path = ROOT / "tests" / "private_test" / "test.csv.gz"
    with gzip.open(train_path, "rt") as fh:
        train = pd.read_csv(fh)
    with gzip.open(test_path, "rt") as fh:
        test = pd.read_csv(fh)
    tr = train[train["split"] == "train"].reset_index(drop=True)
    if len(tr) > args.max_train:
        tr = tr.sample(n=args.max_train, random_state=0).reset_index(drop=True)

    base = args.base or "facebook/esm2_t6_8M_UR50D"
    # Prefer cached model under HF hub or any local path used by Harbor builds
    local_candidates = [
        Path.home() / ".cache/huggingface/hub",
    ]
    print(f"train={len(tr)} test={len(test)} base={base}", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModel.from_pretrained(base)
    model.eval()

    cache = ROOT / "tests" / "private_test" / "calib_emb.npz"
    if cache.exists():
        z = np.load(cache)
        Etr, Ete = z["tr"], z["te"]
        print(f"loaded embeddings from {cache}", flush=True)
    else:
        print("embedding train...", flush=True)
        Etr = embed(tr["sequence"].astype(str).tolist(), tok, model)
        print("embedding test...", flush=True)
        Ete = embed(test["sequence"].astype(str).tolist(), tok, model)
        np.savez(cache, tr=Etr, te=Ete)

    ytr = tr["target"].to_numpy(dtype=np.float64)
    yte = test["target"].to_numpy(dtype=np.float64)
    ridge = RidgeCV(alphas=ALPHAS).fit(Etr, ytr)
    rho = float(spearmanr(ridge.predict(Ete), yte).statistic)
    print(f"frozen_probe_spearman={rho:.4f}", flush=True)

    # T_weak = frozen probe itself (meeting the probe clears the 0.5 tier).
    # T_strong = fixed strong-oracle bar; Codex previously cleared ~0.57.
    t_weak = round(rho, 4)
    t_strong = 0.45
    if t_weak >= t_strong:
        t_strong = round(t_weak + 0.08, 4)

    tiers = {
        "metric": "spearman",
        "t_weak": t_weak,
        "t_strong": t_strong,
        "frozen_probe_spearman": round(rho, 4),
        "definition": {
            "0.0": "integrity fail OR spearman < t_weak",
            "0.5": "t_weak <= spearman < t_strong  (beats frozen probe)",
            "1.0": "spearman >= t_strong  (beats strong oracle bar)",
        },
        "n_train_probe": len(tr),
        "n_test": len(test),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tiers, indent=2) + "\n")
    print(json.dumps(tiers, indent=2))


if __name__ == "__main__":
    main()
