"""Write the rebuilt bbbp private split for the anchor chosen in bbbp_split_v2.json.

Why not write_region_split.py
-----------------------------
That entrypoint indexes anchors against `fetch(name)` -- MoleculeNet's original
row order. The anchor selected by modal_bbbp_split.py is indexed against a
different ordering: the union of the two shipped bbbp CSVs, deduplicated on
smiles and reindexed. Passing 1026 to write_region_split.py would silently hold
out an unrelated region of chemical space. The ordering is reproduced here
exactly instead.

Union-of-halves is also idempotent: re-partitioning does not change the set of
molecules, so this script can be re-run against its own output.

Anchor 1026, selected on separation (band / pooled noise) among candidates with
at least 100 minority-class test molecules, measured at 5 seeds:

    base       0.8934 +/- 0.0028   (encoder frozen, CLS head trained)
    reference  0.9121 +/- 0.0021   (full fine-tune, CLS head)
    band       0.0187 at 5.34 sigma

replacing the rare-scaffold-tail split, which measured band 0.0006 at 0.34 sigma.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

ANCHOR = 1026
TEST_FRAC = 0.2

# Measured by scripts/modal_bbbp_split.py --confirm-anchor 1026 (5 seeds).
ANCHORS = {"base_auc": 0.8934, "reference_auc": 0.9121,
           "band": 0.0187, "band_sigma": 5.34}
EXPECTED = {"n_test": 407, "test_pos_rate": 0.754, "n_minority_in_test": 100}


def morgan_bits(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            X[i] = np.array(gen.GetFingerprint(m), dtype=np.float32)
    return (X > 0).astype(np.float32)


def inchikeys(smiles):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(Chem.MolToInchiKey(m) if m is not None else None)
    return out


def main() -> int:
    agent_csv = HERE / "split" / "agent" / "bbbp_train.csv"
    priv_csv = HERE / "split" / "private" / "bbbp_test.csv"

    # Exactly modal_bbbp_split._all_molecules().
    df = pd.concat([pd.read_csv(agent_csv), pd.read_csv(priv_csv)],
                   ignore_index=True).drop_duplicates("smiles").reset_index(drop=True)
    labels = [c for c in df.columns if c != "smiles"]
    smiles = df["smiles"].tolist()
    n_test = int(TEST_FRAC * len(df))
    print(f"union: {len(df)} molecules, {len(labels)} task(s), n_test={n_test}")

    F = morgan_bits(smiles)
    a = F[ANCHOR]
    inter = F @ a
    sim = inter / np.maximum(F.sum(1) + a.sum() - inter, 1e-9)
    order = np.argsort(-sim)
    test_idx, train_idx = np.sort(order[:n_test]), np.sort(order[n_test:])

    test = df.iloc[test_idx].reset_index(drop=True)
    train = df.iloc[train_idx].reset_index(drop=True)

    # The split must match the partition the anchors were measured on, or the
    # anchors describe a different task than the one that ships.
    Yte = test[labels].to_numpy(float)
    obs = Yte[~np.isnan(Yte)]
    n_pos = int((obs == 1).sum())
    got = {"n_test": len(test), "test_pos_rate": round(n_pos / len(obs), 3),
           "n_minority_in_test": min(n_pos, len(obs) - n_pos)}
    if got != EXPECTED:
        print(f"ABORT: partition does not reproduce the measured one.\n"
              f"  expected {EXPECTED}\n  got      {got}", file=sys.stderr)
        return 1
    print(f"partition reproduces the measured one: {got}")

    train_keys = {k for k in inchikeys(train["smiles"]) if k}
    tk = inchikeys(test["smiles"])
    dupes = [i for i, k in enumerate(tk) if k and k in train_keys]
    if dupes:
        test = test.drop(index=dupes).reset_index(drop=True)
        tk = inchikeys(test["smiles"])
    leaked = sum(1 for k in tk if k and k in train_keys)
    if leaked:
        print(f"ABORT: {leaked} test molecules still in train", file=sys.stderr)
        return 1

    train.to_csv(agent_csv, index=False)
    test.to_csv(priv_csv, index=False)

    manifest_path = HERE / "split" / "private" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["datasets"]["bbbp"] = {
        "dataset": "bbbp",
        "n_tasks": len(labels),
        "labels": labels,
        "split": "Tanimoto anchor-region holdout (private anchor index)",
        "n_train": len(train),
        "n_test": len(test),
        "n_dropped_duplicates": len(dupes),
        "train_test_inchikey_overlap": leaked,
        "test_mean_similarity_to_anchor": round(float(sim[test_idx].mean()), 4),
        "train_pos_rate": round(float(np.nanmean(train[labels].to_numpy(float))), 4),
        "test_pos_rate": round(float(np.nanmean(test[labels].to_numpy(float))), 4),
        "n_minority_in_test": got["n_minority_in_test"],
        "selection_rule": "largest band / pooled noise among candidates with >= 100 "
                          "minority-class test molecules; all survivors fine-tuned",
        "anchors": ANCHORS,
        "supersedes": {
            "design": "rare-scaffold tail (tail_frac 0.35), rejected in region_split.py",
            "measured_band": 0.0006, "measured_band_sigma": 0.34,
        },
    }
    manifest["selection_rule_bbbp"] = manifest["datasets"]["bbbp"]["selection_rule"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    keys_path = HERE / "split" / "private" / "test_inchikeys.json"
    keys = json.loads(keys_path.read_text()) if keys_path.exists() else {}
    keys["bbbp"] = sorted(k for k in tk if k)
    keys_path.write_text(json.dumps(keys, indent=2) + "\n")

    print(f"wrote {agent_csv} ({len(train)} rows)")
    print(f"wrote {priv_csv} ({len(test)} rows, {got['n_minority_in_test']} minority)")
    print(f"updated {manifest_path} and {keys_path}")
    print("private/ and the anchor index must not reach the agent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
