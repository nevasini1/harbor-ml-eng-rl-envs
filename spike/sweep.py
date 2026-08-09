"""Scan Hydro split regimes for one where BOTH pretraining and training matter.

The two published extremes each fail as task designs:
  - random_split : one-hot ridge ties ESM embeddings, so the pretrained model is dead weight.
  - to_P<wt>     : only zero-shot transfers; supervised learning scores below the base model.

A usable task needs a regime where the pretrained model is load-bearing AND a
training loop beats zero-shot. The prime candidate is low-label transfer: train
on the other backbones plus a handful of labelled variants from the target one.

Embeddings are cached per sequence so every regime below is nearly free after
the first pass.
"""

import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from transformers import AutoModelForMaskedLM, AutoTokenizer

DATA = Path(__file__).parent / "data" / "flip2" / "hydro"
CACHE = Path(__file__).parent / "cache"
BASE_MODEL = "facebook/esm2_t6_8M_UR50D"
ALPHAS = np.logspace(-3, 6, 30)
SEED = 0
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_all() -> pd.DataFrame:
    """Every Hydro variant once, with backbone id. random_split covers all rows."""
    with gzip.open(DATA / "random_split.csv.gz", "rt") as fh:
        df = pd.read_csv(fh)
    df["len"] = df["sequence"].str.len()
    df = df.drop(columns=["set", "validation"]).reset_index(drop=True)
    return df


def wt_info(df: pd.DataFrame) -> dict:
    out = {}
    for length, sub in df.groupby("len"):
        cons, var = [], []
        for i in range(length):
            col = Counter(s[i] for s in sub["sequence"])
            cons.append(col.most_common(1)[0][0])
            if len(col) > 1:
                var.append(i)
        out[length] = ("".join(cons), var)
    return out


def n_mutations(df: pd.DataFrame, wts: dict) -> np.ndarray:
    return np.array([
        sum(1 for p in wts[l][1] if s[p] != wts[l][0][p])
        for s, l in zip(df["sequence"], df["len"])
    ])


# --------------------------------------------------------------- feature sets

def feat_onehot(df: pd.DataFrame, wts: dict) -> np.ndarray:
    cols = [(l, p, a) for l, (_, var) in sorted(wts.items()) for p in var for a in "FILMV"]
    index = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(df), len(cols)), dtype=np.float32)
    for r, (seq, l) in enumerate(zip(df["sequence"], df["len"])):
        for p in wts[l][1]:
            X[r, index[(l, p, seq[p])]] = 1.0
    return X


def feat_aacomp(df: pd.DataFrame) -> np.ndarray:
    idx = {a: i for i, a in enumerate(AA)}
    X = np.zeros((len(df), len(AA)), dtype=np.float32)
    for r, seq in enumerate(df["sequence"]):
        for ch in seq:
            if ch in idx:
                X[r, idx[ch]] += 1.0
        X[r] /= max(len(seq), 1)
    return X


# ----------------------------------------------------------------- ESM tables

@torch.no_grad()
def build_esm_tables(df: pd.DataFrame, wts: dict, threads: int):
    torch.set_num_threads(threads)
    emb_path, zs_path = CACHE / "hydro_emb.npy", CACHE / "hydro_zs.npy"
    CACHE.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    model.eval()

    if zs_path.exists():
        zs = np.load(zs_path)
    else:
        lp = {}
        for l, (wt, var) in wts.items():
            for p in var:
                enc = tok([wt], return_tensors="pt")
                enc["input_ids"][0, p + 1] = tok.mask_token_id
                lp[(l, p)] = torch.log_softmax(model(**enc).logits[0, p + 1].double(), -1)
        zs = np.zeros(len(df))
        for r, (seq, l) in enumerate(zip(df["sequence"], df["len"])):
            wt, var = wts[l]
            zs[r] = sum(
                float(lp[(l, p)][tok.convert_tokens_to_ids(seq[p])])
                - float(lp[(l, p)][tok.convert_tokens_to_ids(wt[p])])
                for p in var if seq[p] != wt[p]
            )
        np.save(zs_path, zs)

    if emb_path.exists():
        emb = np.load(emb_path)
    else:
        t0, seqs, out = time.time(), df["sequence"].tolist(), []
        for i in range(0, len(seqs), 64):
            batch = seqs[i : i + 64]
            enc = tok(batch, return_tensors="pt", padding=True)
            hidden = model.esm(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mask[:, 0] = 0
            for j, s in enumerate(batch):
                mask[j, len(s) + 1] = 0
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
        emb = np.concatenate(out).astype(np.float32)
        np.save(emb_path, emb)
        print(f"[embedded {len(seqs):,} seqs in {time.time()-t0:.0f}s]")

    return emb, zs


# ------------------------------------------------------------------- regimes

def evaluate(tr_idx, te_idx, feats: dict, y, zs) -> dict:
    scores = {}
    for name, X in feats.items():
        m = RidgeCV(alphas=ALPHAS).fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        rho = spearmanr(pred, y[te_idx]).statistic
        scores[name] = round(float(rho) if np.isfinite(rho) else 0.0, 4)
    scores["zeroshot"] = round(float(spearmanr(zs[te_idx], y[te_idx]).statistic), 4)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shots", nargs="+", type=int, default=[0, 25, 50, 100, 200, 500, 1000])
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df = load_all()
    wts = wt_info(df)
    y = df["target"].to_numpy()
    nmut = n_mutations(df, wts)
    emb, zs = build_esm_tables(df, wts, args.threads)

    feats = {
        "onehot": feat_onehot(df, wts),
        "aa-comp": feat_aacomp(df),
        "esm-emb": emb,
        "esm-emb+zs": np.hstack([emb, zs.reshape(-1, 1).astype(np.float32)]),
    }

    results = []

    print("\n### regime: low-label transfer to a held-out backbone")
    print("(train = other two backbones + k labelled target variants; test = rest of target)")
    for target in sorted(wts):
        others = np.where(df["len"].to_numpy() != target)[0]
        tgt = np.where(df["len"].to_numpy() == target)[0]
        for k in args.shots:
            shot = rng.choice(tgt, size=k, replace=False) if k else np.array([], dtype=int)
            te = np.setdiff1d(tgt, shot)
            tr = np.concatenate([others, shot]).astype(int)
            s = evaluate(tr, te, feats, y, zs)
            s |= {"regime": "low-n-transfer", "target_len": int(target), "k": k,
                  "n_train": len(tr), "n_test": len(te)}
            results.append(s)
            print(f"  wt_len={target} k={k:<5} n_test={len(te):<6} "
                  + "  ".join(f"{n}={s[n]:+.3f}" for n in
                              ["onehot", "aa-comp", "esm-emb", "esm-emb+zs", "zeroshot"]))

    print("\n### regime: mutation-order extrapolation, within backbone")
    for target in sorted(wts):
        tgt = df["len"].to_numpy() == target
        for cut in (3, 4):
            tr = np.where(tgt & (nmut <= cut))[0]
            te = np.where(tgt & (nmut > cut))[0]
            s = evaluate(tr, te, feats, y, zs)
            s |= {"regime": "mut-order", "target_len": int(target), "cut": cut,
                  "n_train": len(tr), "n_test": len(te)}
            results.append(s)
            print(f"  wt_len={target} train<={cut}  n_train={len(tr):<6} n_test={len(te):<6} "
                  + "  ".join(f"{n}={s[n]:+.3f}" for n in
                              ["onehot", "aa-comp", "esm-emb", "esm-emb+zs", "zeroshot"]))

    out = Path(__file__).parent / "results" / "sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
