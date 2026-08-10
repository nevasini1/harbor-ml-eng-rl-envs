"""Anchor measurements on FLIP2 Meltome-mixed.

Hydro failed Gate A structurally: it is a combinatorial library over 7 fixed
positions, so a 35-feature one-hot ridge is the ceiling and the pretrained model
is dead weight. Meltome-mixed is the opposite regime - 23k distinct proteins with
no shared coordinate system - so position one-hot is not even definable and any
signal must come from sequence modelling.

Controls here are the strongest classical featurizations that do transfer across
unrelated sequences: amino-acid composition, k-mer counts, and length.
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from transformers import AutoModelForMaskedLM, AutoTokenizer

DATA = Path(__file__).parent / "data" / "flip2" / "meltome"
CACHE = Path(__file__).parent / "cache"
BASE_MODEL = "facebook/esm2_t6_8M_UR50D"
ALPHAS = np.logspace(-3, 6, 30)
AA = "ACDEFGHIKLMNPQRSTVWY"
SEED = 0


def load(max_len: int) -> pd.DataFrame:
    with gzip.open(DATA / "mixed_split.csv.gz", "rt") as fh:
        df = pd.read_csv(fh)
    df["raw_len"] = df["sequence"].str.len()
    df["sequence"] = df["sequence"].str.slice(0, max_len)
    return df.reset_index(drop=True)


def feat_aacomp(seqs) -> np.ndarray:
    idx = {a: i for i, a in enumerate(AA)}
    X = np.zeros((len(seqs), len(AA) + 1), dtype=np.float32)
    for r, s in enumerate(seqs):
        for ch in s:
            if ch in idx:
                X[r, idx[ch]] += 1.0
        X[r, : len(AA)] /= max(len(s), 1)
        X[r, len(AA)] = np.log1p(len(s))
    return X


def kmer_ridge(seqs, tr, te, y, k: int = 3, top: int = 2000) -> float:
    """k-mer control, kept sparse.

    Dense RidgeCV on this matrix is a single-threaded SVD that ran for 50+ minutes,
    so select alpha on a held-out slice of train with the sparse CG solver instead.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    vec = TfidfVectorizer(analyzer="char", ngram_range=(k, k), max_features=top)
    X = vec.fit_transform(list(seqs))

    tr_idx = np.where(tr)[0]
    rng = np.random.default_rng(SEED)
    rng.shuffle(tr_idx)
    cut = int(0.85 * len(tr_idx))
    fit_idx, val_idx = tr_idx[:cut], tr_idx[cut:]

    best = (-2.0, None)
    for alpha in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        m = Ridge(alpha=alpha, solver="sparse_cg", max_iter=2000).fit(X[fit_idx], y[fit_idx])
        rho = spearmanr(m.predict(X[val_idx]), y[val_idx]).statistic
        if rho > best[0]:
            best = (rho, alpha)
    m = Ridge(alpha=best[1], solver="sparse_cg", max_iter=2000).fit(X[tr], y[tr])
    return float(spearmanr(m.predict(X[te]), y[te]).statistic)


@torch.no_grad()
def esm_embed(seqs, threads: int, tag: str) -> np.ndarray:
    path = CACHE / f"meltome_emb_{tag}.npy"
    if path.exists():
        return np.load(path)

    CACHE.mkdir(exist_ok=True)
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    model.eval()

    order = np.argsort([len(s) for s in seqs])  # length-sorted batching
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    bs = 32
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        batch = [seqs[j] for j in idx]
        enc = tok(batch, return_tensors="pt", padding=True)
        hidden = model.esm(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mask[:, 0] = 0
        for j, s in enumerate(batch):
            mask[j, len(s) + 1] = 0
        out[idx] = ((hidden * mask).sum(1) / mask.sum(1)).numpy()
        if i % (bs * 100) == 0:
            done = i + len(idx)
            print(f"    embedded {done:,}/{len(seqs):,} ({done/max(time.time()-t0,1e-9):.0f} seq/s)")
    print(f"    [embedding took {time.time()-t0:.0f}s = {len(seqs)/(time.time()-t0):.0f} seq/s]")
    np.save(path, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    df = load(args.max_len)
    tr = (df["set"] == "train").to_numpy()
    te = (df["set"] == "test").to_numpy()
    y = df["target"].to_numpy()
    seqs = df["sequence"].tolist()

    print(f"meltome-mixed  train={tr.sum():,}  test={te.sum():,}  max_len={args.max_len}")
    print(f"raw length: median={int(df['raw_len'].median())} "
          f"p95={int(df['raw_len'].quantile(0.95))} max={df['raw_len'].max():,} "
          f"(truncated to {args.max_len})")

    results = {}

    def record(name, rho, secs):
        results[name] = {"spearman": round(float(rho), 4), "seconds": round(secs, 1)}
        print(f"  {name:<14} spearman={rho:+.4f}  ({secs:.1f}s)", flush=True)

    # ESM first: it is the gating number and the cache makes reruns cheap.
    t0 = time.time()
    E = esm_embed(seqs, args.threads, f"len{args.max_len}")
    embed_secs = time.time() - t0
    t0 = time.time()
    record("frozen-ridge", spearmanr(RidgeCV(alphas=ALPHAS).fit(E[tr], y[tr]).predict(E[te]), y[te]).statistic, time.time() - t0)

    t0 = time.time()
    X = feat_aacomp(seqs)
    record("aa-comp", spearmanr(RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr]).predict(X[te]), y[te]).statistic, time.time() - t0)

    t0 = time.time()
    Xc = np.hstack([E, X])
    record("esm+aacomp", spearmanr(RidgeCV(alphas=ALPHAS).fit(Xc[tr], y[tr]).predict(Xc[te]), y[te]).statistic, time.time() - t0)

    t0 = time.time()
    record("3mer-tfidf", kmer_ridge(seqs, tr, te, y), time.time() - t0)

    results["_embed_seconds"] = round(embed_secs, 1)
    out = Path(__file__).parent / "results" / f"meltome_len{args.max_len}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
