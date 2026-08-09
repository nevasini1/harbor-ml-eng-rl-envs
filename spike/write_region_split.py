"""Write the final private region-holdout split for the chosen anchors.

Selection rule, fixed in advance and recorded in the manifest: among candidate
anchors where every task remains scoreable in the holdout, take the one with the
LOWEST frozen-probe AUC, i.e. the hardest region. Selecting for difficulty buys
measurement range and is independent of any particular agent strategy.

The anchor index and the seed that generated the candidate list are private.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_private_split import inchikeys
from moleculenet import fetch, morgan
from region_split import tanimoto_to

HERE = Path(__file__).parent
OUT = HERE / "split"


def build(name: str, anchor: int, test_frac: float) -> dict:
    df, labels, digest = fetch(name)
    smiles = df["smiles"].tolist()
    F = (morgan(smiles) > 0).astype(np.float32)
    sim = tanimoto_to(F, F[anchor])
    order = np.argsort(-sim)
    n_test = int(test_frac * len(df))

    test = df.iloc[np.sort(order[:n_test])].reset_index(drop=True)
    train = df.iloc[np.sort(order[n_test:])].reset_index(drop=True)

    train_keys = set(k for k in inchikeys(train["smiles"]) if k)
    tk = inchikeys(test["smiles"])
    dupes = [i for i, k in enumerate(tk) if k and k in train_keys]
    if dupes:
        test = test.drop(index=dupes).reset_index(drop=True)
        tk = inchikeys(test["smiles"])
    leaked = sum(1 for k in tk if k and k in train_keys)

    (OUT / "agent").mkdir(parents=True, exist_ok=True)
    (OUT / "private").mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "agent" / f"{name}_train.csv", index=False)
    test.to_csv(OUT / "private" / f"{name}_test.csv", index=False)

    Y = df[labels].to_numpy(dtype=np.float64)
    info = {
        "dataset": name, "source_sha256": digest, "n_tasks": len(labels),
        "labels": labels, "n_train": len(train), "n_test": len(test),
        "n_dropped_duplicates": len(dupes), "train_test_inchikey_overlap": leaked,
        "test_mean_similarity_to_anchor": round(float(sim[order[:n_test]].mean()), 4),
        "train_pos_rate": round(float(np.nanmean(train[labels].to_numpy(float))), 4),
        "test_pos_rate": round(float(np.nanmean(test[labels].to_numpy(float))), 4),
    }
    print(f"{name:<8} train={len(train):>6,} test={len(test):>6,} "
          f"dropped={len(dupes)} overlap={leaked} "
          f"test_pos={info['test_pos_rate']:.3f} train_pos={info['train_pos_rate']:.3f}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", nargs="+", required=True,
                    help="dataset=anchor_index pairs, e.g. tox21=3357")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    chosen = dict(p.split("=") for p in args.anchors)
    manifest = {"test_frac": args.test_frac, "selection_rule":
                "lowest frozen-probe AUC among fully-scoreable candidates",
                "datasets": {}}
    keys = {}
    for name, anchor in chosen.items():
        info = build(name, int(anchor), args.test_frac)
        manifest["datasets"][name] = info
        te = pd.read_csv(OUT / "private" / f"{name}_test.csv")
        keys[name] = sorted(k for k in inchikeys(te["smiles"]) if k)

    (OUT / "private" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "private" / "test_inchikeys.json").write_text(json.dumps(keys, indent=2))
    print(f"\nwrote {OUT}. private/ and the anchor indices must not reach the agent.")


if __name__ == "__main__":
    main()
