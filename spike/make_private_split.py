"""Construct the private scaffold split for BBBP and Tox21.

The published MoleculeNet protocol fills train/valid/test from the largest
Bemis-Murcko scaffold groups downward, which is fully deterministic and therefore
reproducible by the agent. We instead partition whole scaffold *groups* at random
under a seed that is never published, so the exact partition cannot be regenerated.

Splitting at group level (not molecule level) preserves the property that matters:
test molecules have structural cores absent from training.

Outputs:
  agent/<ds>_train.csv    - given to the agent, labels included
  private/<ds>_test.csv   - verifier only, never enters the agent container
  private/manifest.json   - InChIKeys of test molecules, for the grade-time
                            train/test overlap check

The seed lives in PRIVATE_SEED (gitignored, verifier build context only). It must
not appear in instruction.md, the agent Dockerfile, or any published artifact.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from moleculenet import DATASETS, fetch

HERE = Path(__file__).parent
OUT = HERE / "split"
EVAL_SETS = ["bbbp", "tox21"]


def scaffold_groups(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    groups = defaultdict(list)
    bad = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append(i)
            continue
        groups[MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)].append(i)
    return groups, bad


def inchikeys(smiles):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    keys = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        keys.append(Chem.MolToInchiKey(mol) if mol is not None else "")
    return keys


def build(name: str, seed: int, test_frac: float, tail_frac: float = 0.35) -> dict:
    """Difficulty-preserving private split.

    A naive random shuffle over scaffold groups makes the task far EASIER than the
    published protocol - measured: BBBP frozen-probe AUC 0.92 vs 0.73 - because the
    published protocol deliberately sends the *rarest* scaffolds to test while a
    random draw sends typical ones.

    So we keep the published difficulty profile and privatise only the selection:
    the largest scaffold groups always go to train, and the private seed chooses
    which of the many rare tail groups become test. With 763 singleton scaffolds in
    BBBP and 1,772 in tox21, that selection is not guessable.
    """
    df, labels, digest = fetch(name)
    smiles = df["smiles"].tolist()
    groups, bad = scaffold_groups(smiles)

    n_total = sum(len(v) for v in groups.values())
    # Deterministic: biggest groups first, ties broken by scaffold string.
    ordered = sorted(groups, key=lambda k: (-len(groups[k]), k))

    head_target = int((1.0 - tail_frac) * n_total)
    train_idx, tail_keys, filled = [], [], 0
    for k in ordered:
        if filled < head_target:
            train_idx += groups[k]
            filled += len(groups[k])
        else:
            tail_keys.append(k)

    rng = np.random.default_rng(seed)
    rng.shuffle(tail_keys)

    target = int(test_frac * n_total)
    test_idx, n_test_groups = [], 0
    for k in tail_keys:
        if len(test_idx) < target:
            test_idx += groups[k]
            n_test_groups += 1
        else:
            train_idx += groups[k]

    train = df.iloc[sorted(train_idx)].reset_index(drop=True)
    test = df.iloc[sorted(test_idx)].reset_index(drop=True)

    # MoleculeNet contains the same compound under different SMILES strings, which
    # can land the same molecule on both sides of a scaffold split. Drop those from
    # test: identity leakage would inflate the score regardless of scaffolds.
    train_keys = set(k for k in inchikeys(train["smiles"]) if k)
    test_keys_all = inchikeys(test["smiles"])
    dupes = [i for i, k in enumerate(test_keys_all) if k and k in train_keys]
    if dupes:
        test = test.drop(index=dupes).reset_index(drop=True)
    test_keys = inchikeys(test["smiles"])
    leaked = sum(1 for k in test_keys if k and k in train_keys)

    (OUT / "agent").mkdir(parents=True, exist_ok=True)
    (OUT / "private").mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "agent" / f"{name}_train.csv", index=False)
    test.to_csv(OUT / "private" / f"{name}_test.csv", index=False)

    info = {
        "dataset": name,
        "source_sha256": digest,
        "n_total": len(df),
        "n_unparseable": len(bad),
        "n_tasks": len(labels),
        "labels": labels,
        "n_scaffold_groups": len(groups),
        "n_test_groups": n_test_groups,
        "n_tail_groups": len(tail_keys),
        "tail_frac": tail_frac,
        "n_train": len(train),
        "n_test": len(test),
        "n_test_dropped_as_duplicates": len(dupes),
        "test_inchikeys_sha256": hashlib.sha256(
            "\n".join(sorted(test_keys)).encode()).hexdigest(),
        "train_test_inchikey_overlap": leaked,
    }
    print(f"{name:<8} train={len(train):>6,} test={len(test):>6,} "
          f"groups={len(groups):>5,} tail={len(tail_keys):>5,} "
          f"test_groups={n_test_groups:>5,} overlap={leaked}")
    return info, test_keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="private seed; defaults to the value in PRIVATE_SEED")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    seed_file = HERE / "PRIVATE_SEED"
    if args.seed is not None:
        seed = args.seed
        seed_file.write_text(str(seed))
    elif seed_file.exists():
        seed = int(seed_file.read_text().strip())
    else:
        raise SystemExit("no --seed given and no PRIVATE_SEED file present")

    OUT.mkdir(exist_ok=True)
    manifest = {"test_frac": args.test_frac, "datasets": {}}
    all_keys = {}
    for name in EVAL_SETS:
        info, keys = build(name, seed, args.test_frac)
        manifest["datasets"][name] = info
        all_keys[name] = sorted(k for k in keys if k)

    (OUT / "private" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "private" / "test_inchikeys.json").write_text(json.dumps(all_keys, indent=2))
    print(f"\nwrote {OUT}/agent and {OUT}/private")
    print("NOTE: split/private and PRIVATE_SEED must never enter the agent container.")


if __name__ == "__main__":
    main()
