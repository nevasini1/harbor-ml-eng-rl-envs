"""Build agent-visible train data and a verifier-only private test split.

Source: FLIP2 Meltome-mixed (Zenodo 18433203, CC-BY 4.0).
We do NOT reuse the published FLIP2 train/test column. Instead we draw a fresh
partition with a secret seed so the held-out set is not identical to any public
FLIP2 split file.

Contamination note (honest): Meltome labels are public CC-BY, so an agent with
network could try to re-derive labels by sequence match. Mitigations are
enforcement: private_test never enters the agent image, verifier is
network_mode=no-network, and grade.py rejects exact train/test sequence overlap
if a train artifact is provided.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Secret seed for the private partition. Do NOT copy this file into the agent
# image. Changing the seed changes the held-out set.
PRIVATE_SEED = "mleval-tasks/sciml-protein-regression/v1/meltome-private-2026-08-08"
TEST_FRACTION = 0.15
VAL_FRACTION_OF_TRAIN = 0.10
MAX_LEN = 512  # truncate for CPU feasibility; grader uses the same rule


def _bucket(seq: str) -> float:
    h = hashlib.sha256(f"{PRIVATE_SEED}:{seq}".encode()).hexdigest()
    return int(h[:16], 16) / 16**16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "research/data/flip2/meltome/mixed_split.csv.gz",
    )
    ap.add_argument(
        "--task-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = ap.parse_args()

    with gzip.open(args.src, "rt") as fh:
        df = pd.read_csv(fh)

    # Dedup by sequence (keep first label); truncate for CPU task contract.
    df = df.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    df["sequence"] = df["sequence"].astype(str).str.slice(0, MAX_LEN)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    scores = df["sequence"].map(_bucket).to_numpy()
    is_test = scores < TEST_FRACTION
    train_pool = df.loc[~is_test].copy()
    test = df.loc[is_test].copy()

    # Carve a small validation slice from train for the agent (visible).
    rng = np.random.default_rng(int(hashlib.sha256(PRIVATE_SEED.encode()).hexdigest()[:8], 16))
    n_val = max(1, int(len(train_pool) * VAL_FRACTION_OF_TRAIN))
    val_idx = set(rng.choice(train_pool.index.to_numpy(), size=n_val, replace=False))
    train_pool["split"] = ["val" if i in val_idx else "train" for i in train_pool.index]
    test["split"] = "test"

    agent_df = train_pool[["sequence", "target", "split"]].reset_index(drop=True)
    private_df = test[["sequence", "target"]].reset_index(drop=True)

    agent_dir = args.task_root / "environment" / "data"
    private_dir = args.task_root / "tests" / "private_test"
    agent_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    agent_path = agent_dir / "train.csv.gz"
    private_path = private_dir / "test.csv.gz"
    with gzip.open(agent_path, "wt") as fh:
        agent_df.to_csv(fh, index=False)
    with gzip.open(private_path, "wt") as fh:
        private_df.to_csv(fh, index=False)

    # Manifest for the verifier image (no seed).
    manifest = {
        "source": "FLIP2 Meltome-mixed (Zenodo 10.5281/zenodo.18433203)",
        "license": "CC-BY-4.0",
        "max_len": MAX_LEN,
        "test_fraction": TEST_FRACTION,
        "n_agent_rows": int(len(agent_df)),
        "n_agent_train": int((agent_df["split"] == "train").sum()),
        "n_agent_val": int((agent_df["split"] == "val").sum()),
        "n_private_test": int(len(private_df)),
        "agent_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest(),
        "private_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
        "overlap_sequences": int(
            len(set(agent_df["sequence"]) & set(private_df["sequence"]))
        ),
        "note": "Partition is a fresh unpublished resplit; seed not shipped to agent.",
    }
    (private_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Agent-visible copy of counts only (no private hashes required, but harmless).
    (agent_dir / "manifest.json").write_text(
        json.dumps(
            {
                "n_train": manifest["n_agent_train"],
                "n_val": manifest["n_agent_val"],
                "max_len": MAX_LEN,
                "task": "predict Meltome Tm (Celsius); higher Spearman is better",
            },
            indent=2,
        )
        + "\n"
    )

    print(json.dumps(manifest, indent=2))
    if manifest["overlap_sequences"] != 0:
        raise SystemExit("BUG: train/test sequence overlap")


if __name__ == "__main__":
    main()
