"""Private chemical-region holdout split, selected on measured difficulty.

Why not a scaffold split. Two attempts failed:

  * Random scaffold-group shuffle - BBBP frozen-probe AUC 0.921 (published: 0.726).
  * Rare-scaffold tail, privately selected - AUC 0.967. Even easier.

Diagnosis: the published MoleculeNet BBBP scaffold split is hard mostly because of a
label-distribution artifact, not scaffold novelty. Its train split is 82% positive and
its test split 53% positive. Reproducing that difficulty means reproducing the
deterministic published partition, which the agent can trivially regenerate.

So difficulty is constructed deliberately instead. A private anchor molecule defines a
contiguous region of chemical space by Tanimoto similarity; the whole region is held
out. Training therefore never sees that region, which is a genuine covariate shift, and
the held-out region is a real generalization target rather than a sampling artifact.

Candidate anchors are screened on measured criteria and the choice is recorded, so the
split is selected transparently rather than tuned until the numbers look nice.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moleculenet import fetch, morgan, multitask_auc, fit_predict

HERE = Path(__file__).parent
BASE_MODEL = "DeepChem/ChemBERTa-77M-MLM"


def tanimoto_to(fps: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    inter = fps @ anchor
    union = fps.sum(1) + anchor.sum() - inter
    return inter / np.maximum(union, 1e-9)


def embed(smiles, threads: int, tag: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache = HERE / "cache" / f"rs_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    cache.parent.mkdir(exist_ok=True)
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), 64):
            enc = tok(list(smiles[i : i + 64]), return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
    emb = np.concatenate(out).astype(np.float32)
    np.save(cache, emb)
    return emb


def evaluate_candidate(E, Y, test_idx, train_idx):
    """Frozen-probe AUC under this partition: the base anchor, and the difficulty proxy."""
    from sklearn.linear_model import LogisticRegression

    try:
        P = fit_predict(lambda: LogisticRegression(max_iter=3000),
                        E[train_idx], Y[train_idx], E[test_idx])
        return multitask_auc(Y[test_idx], P)
    except Exception:
        return float("nan")


def scoreable_tasks(Y, idx) -> int:
    n = 0
    for t in range(Y.shape[1]):
        m = ~np.isnan(Y[idx, t])
        if m.sum() >= 20 and len(np.unique(Y[idx, t][m])) == 2:
            n += 1
    return n


def screen(name: str, seed: int, test_frac: float, threads: int, n_candidates: int):
    df, labels, digest = fetch(name)
    smiles = df["smiles"].tolist()
    Y = df[labels].to_numpy(dtype=np.float64)
    F = (morgan(smiles) > 0).astype(np.float32)
    E = embed(smiles, threads, name)
    n_test = int(test_frac * len(df))

    rng = np.random.default_rng(seed)
    candidates = rng.choice(len(df), size=n_candidates, replace=False)

    rows = []
    for anchor_i in candidates:
        sim = tanimoto_to(F, F[anchor_i])
        order = np.argsort(-sim)
        test_idx = np.sort(order[:n_test])
        train_idx = np.sort(order[n_test:])
        n_ok = scoreable_tasks(Y, test_idx)
        if n_ok < max(1, Y.shape[1] // 2):
            continue
        auc = evaluate_candidate(E, Y, test_idx, train_idx)
        pos = np.nanmean(Y[test_idx]) if Y.shape[1] == 1 else np.nanmean(Y[test_idx])
        rows.append({
            "anchor": int(anchor_i),
            "frozen_probe_auc": round(float(auc), 4),
            "n_scoreable_tasks": n_ok,
            "test_pos_rate": round(float(pos), 4),
            "train_pos_rate": round(float(np.nanmean(Y[train_idx])), 4),
            "mean_sim_in_test": round(float(sim[test_idx].mean()), 4),
        })
    rows.sort(key=lambda r: r["frozen_probe_auc"])
    return df, labels, digest, Y, rows, n_test, F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["bbbp", "tox21"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=12)
    args = ap.parse_args()

    seed_file = HERE / "PRIVATE_SEED"
    seed = args.seed if args.seed is not None else int(seed_file.read_text().strip())

    report = {}
    for name in args.datasets:
        df, labels, digest, Y, rows, n_test, F = screen(
            name, seed, args.test_frac, args.threads, args.candidates)
        print(f"\n=== {name}: {len(df):,} molecules, {len(labels)} task(s), "
              f"holdout {n_test:,} ===")
        print(f"{'anchor':>7} {'probe AUC':>10} {'tasks':>6} {'test pos':>9} "
              f"{'train pos':>10} {'mean sim':>9}")
        for r in rows:
            print(f"{r['anchor']:>7} {r['frozen_probe_auc']:>10.4f} "
                  f"{r['n_scoreable_tasks']:>6} {r['test_pos_rate']:>9.3f} "
                  f"{r['train_pos_rate']:>10.3f} {r['mean_sim_in_test']:>9.3f}")
        report[name] = rows

    dest = HERE / "results" / "region_candidates.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
