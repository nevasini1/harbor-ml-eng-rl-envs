"""Measure the four anchor points on FLIP2 Hydro splits.

Anchors, weakest to strongest:
  1. one-hot        - per-(wild-type, position, residue) indicator + ridge. No pretraining.
  2. aa-comp        - amino-acid composition + ridge. No pretraining, transfers across backbones.
  3. zeroshot       - ESM-2-8M masked-marginal pseudo-likelihood. Pretraining, no training.
  4. frozen-ridge   - ESM-2-8M mean-pooled embeddings + ridge. Pretraining, cheap head only.
  5. finetune       - full fine-tune of EsmForSequenceClassification. The reference anchor.

Gate A asks whether (5) clears (4) and whether (4) clears (1)/(2). If a trivial
one-hot ridge matches the fine-tune, the task has no room for an agent to work in.
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


def load_split(split: str) -> pd.DataFrame:
    with gzip.open(DATA / f"{split}.csv.gz", "rt") as fh:
        df = pd.read_csv(fh)
    df["len"] = df["sequence"].str.len()
    return df


def wt_info(df: pd.DataFrame) -> dict[int, tuple[str, list[int]]]:
    """Consensus wild-type and variable positions per backbone (keyed by length)."""
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


# ---------------------------------------------------------------- featurizers

def feat_onehot(df: pd.DataFrame, wts: dict) -> np.ndarray:
    cols = []
    for length, (_, var) in sorted(wts.items()):
        for pos in var:
            for aa in "FILMV":
                cols.append((length, pos, aa))
    index = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(df), len(cols)), dtype=np.float32)
    for row, (seq, length) in enumerate(zip(df["sequence"], df["len"])):
        for pos in wts[length][1]:
            key = (length, pos, seq[pos])
            if key in index:
                X[row, index[key]] = 1.0
    return X


AA = "ACDEFGHIKLMNPQRSTVWY"


def feat_aacomp(df: pd.DataFrame, wts: dict) -> np.ndarray:
    idx = {a: i for i, a in enumerate(AA)}
    X = np.zeros((len(df), len(AA)), dtype=np.float32)
    for row, seq in enumerate(df["sequence"]):
        for ch in seq:
            if ch in idx:
                X[row, idx[ch]] += 1.0
        X[row] /= max(len(seq), 1)
    return X


# ------------------------------------------------------------------- ESM bits

def load_esm():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)
    model.eval()
    return tok, model


@torch.no_grad()
def esm_embeddings(seqs: list[str], tok, model, batch_size: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True)
        hidden = model.esm(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # drop BOS/EOS from the mean
        mask[:, 0] = 0
        for j, s in enumerate(batch):
            mask[j, len(s) + 1] = 0
        out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def esm_zeroshot(df: pd.DataFrame, wts: dict, tok, model) -> np.ndarray:
    """Masked-marginal score: sum over mutated positions of log p(mut) - log p(wt)."""
    logprob_cache: dict[tuple[int, int], torch.Tensor] = {}
    for length, (wt, var) in wts.items():
        for pos in var:
            enc = tok([wt], return_tensors="pt")
            # +1 for the BOS token ESM prepends
            enc["input_ids"][0, pos + 1] = tok.mask_token_id
            logits = model(**enc).logits[0, pos + 1]
            logprob_cache[(length, pos)] = torch.log_softmax(logits.double(), dim=-1)

    scores = np.zeros(len(df), dtype=np.float64)
    for row, (seq, length) in enumerate(zip(df["sequence"], df["len"])):
        wt, var = wts[length]
        total = 0.0
        for pos in var:
            if seq[pos] != wt[pos]:
                lp = logprob_cache[(length, pos)]
                total += float(lp[tok.convert_tokens_to_ids(seq[pos])])
                total -= float(lp[tok.convert_tokens_to_ids(wt[pos])])
        scores[row] = total
    return scores


# ----------------------------------------------------------------- evaluation

def ridge_eval(Xtr, ytr, Xte, yte) -> float:
    model = RidgeCV(alphas=ALPHAS)
    model.fit(Xtr, ytr)
    return float(spearmanr(model.predict(Xte), yte).statistic)


def run(split: str, do_finetune: bool, epochs: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = load_split(split)
    wts = wt_info(df)
    tr = df[df["set"] == "train"].reset_index(drop=True)
    te = df[df["set"] == "test"].reset_index(drop=True)
    ytr, yte = tr["target"].to_numpy(), te["target"].to_numpy()

    print(f"\n{'='*70}\nsplit={split}  train={len(tr):,}  test={len(te):,}  threads={threads}")
    print(f"train backbones={sorted(tr['len'].unique())}  test backbones={sorted(te['len'].unique())}")
    results: dict[str, dict] = {}

    def record(name: str, rho: float, secs: float) -> None:
        results[name] = {"spearman": round(rho, 4), "seconds": round(secs, 1)}
        print(f"  {name:<14} spearman={rho:+.4f}   ({secs:.1f}s)")

    t0 = time.time()
    record("onehot", ridge_eval(feat_onehot(tr, wts), ytr, feat_onehot(te, wts), yte), time.time() - t0)

    t0 = time.time()
    record("aa-comp", ridge_eval(feat_aacomp(tr, wts), ytr, feat_aacomp(te, wts), yte), time.time() - t0)

    tok, model = load_esm()

    t0 = time.time()
    record("zeroshot", float(spearmanr(esm_zeroshot(te, wts, tok, model), yte).statistic), time.time() - t0)

    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    key = CACHE / f"emb_{split}.npz"
    if key.exists():
        z = np.load(key)
        Etr, Ete = z["tr"], z["te"]
        cached = True
    else:
        Etr = esm_embeddings(tr["sequence"].tolist(), tok, model)
        Ete = esm_embeddings(te["sequence"].tolist(), tok, model)
        np.savez(key, tr=Etr, te=Ete)
        cached = False
    embed_secs = time.time() - t0
    print(f"  [embeddings {'cached' if cached else 'computed'} in {embed_secs:.1f}s "
          f"for {len(tr)+len(te):,} seqs = {(len(tr)+len(te))/max(embed_secs,1e-9):.0f} seq/s]")
    t0 = time.time()
    record("frozen-ridge", ridge_eval(Etr, ytr, Ete, yte), time.time() - t0)

    if do_finetune:
        rho, secs, curve = finetune(tr, te, ytr, yte, epochs, threads)
        record(f"finetune-e{epochs}", rho, secs)
        results[f"finetune-e{epochs}"]["curve"] = curve

    return {"split": split, "n_train": len(tr), "n_test": len(te), "anchors": results}


def finetune(tr, te, ytr, yte, epochs: int, threads: int):
    from transformers import EsmForSequenceClassification

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = EsmForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.train()

    mu, sd = ytr.mean(), ytr.std()
    ytr_z = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    seqs = tr["sequence"].tolist()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bs = 32
    curve = []
    t0 = time.time()

    for ep in range(epochs):
        perm = torch.randperm(len(seqs))
        model.train()
        for i in range(0, len(seqs), bs):
            idx = perm[i : i + bs].tolist()
            enc = tok([seqs[j] for j in idx], return_tensors="pt", padding=True)
            pred = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, ytr_z[idx])
            loss.backward()
            opt.step()
            opt.zero_grad()
        rho = predict_spearman(model, tok, te["sequence"].tolist(), yte)
        curve.append(round(rho, 4))
        print(f"    epoch {ep+1}/{epochs}: test spearman={rho:+.4f}  "
              f"({time.time()-t0:.0f}s elapsed, {(ep+1)*len(seqs)/(time.time()-t0):.0f} seq/s)")

    return curve[-1], time.time() - t0, curve


@torch.no_grad()
def predict_spearman(model, tok, seqs, y, bs: int = 64) -> float:
    model.eval()
    preds = []
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i : i + bs], return_tensors="pt", padding=True)
        preds.append(model(**enc).logits.squeeze(-1).numpy())
    return float(spearmanr(np.concatenate(preds), y).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["random_split", "to_P06241"])
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/anchors.json")
    args = ap.parse_args()

    all_results = [run(s, args.finetune, args.epochs, args.threads) for s in args.splits]
    out = Path(__file__).parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
