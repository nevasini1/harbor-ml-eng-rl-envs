"""Reverse-engineer Hydro's structure so we can design a private position split.

FLIP2 ships only (sequence, target, set, validation) with no wild-type or
position annotation, so we recover the wild-type backbones and the randomized
core positions ourselves by grouping on length and taking per-position consensus.
"""

import gzip
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "flip2" / "hydro"


def load(split: str) -> pd.DataFrame:
    with gzip.open(DATA / f"{split}.csv.gz", "rt") as fh:
        return pd.read_csv(fh)


def variable_positions(seqs: pd.Series) -> tuple[str, list[int]]:
    """Consensus (pseudo wild-type) plus positions that vary across the group."""
    length = len(seqs.iloc[0])
    consensus = []
    variable = []
    for i in range(length):
        col = Counter(s[i] for s in seqs)
        consensus.append(col.most_common(1)[0][0])
        if len(col) > 1:
            variable.append(i)
    return "".join(consensus), variable


def main() -> None:
    df = load("random_split")
    df["len"] = df["sequence"].str.len()

    print("=== length groups (candidate wild-type backbones) ===")
    print(df["len"].value_counts().sort_index().to_string())

    groups = {}
    for length, sub in df.groupby("len"):
        wt, var = variable_positions(sub["sequence"])
        groups[length] = (wt, var, len(sub))
        print(f"\n--- length {length}: {len(sub):,} sequences ---")
        print(f"  variable positions ({len(var)}): {var}")
        print(f"  consensus/WT: {wt}")
        n_mut = sub["sequence"].apply(
            lambda s: sum(1 for i in var if s[i] != wt[i])
        )
        print(f"  mutations per seq: {n_mut.value_counts().sort_index().to_dict()}")
        print(f"  target: min={sub['target'].min():.3f} "
              f"median={sub['target'].median():.3f} max={sub['target'].max():.3f}")
        aa_by_pos = {i: sorted({s[i] for s in sub['sequence']}) for i in var}
        for i, aas in aa_by_pos.items():
            print(f"    pos {i:>3}: {len(aas)} aa -> {''.join(aas)}")

    print("\n=== cross-check: how the published WT splits partition the length groups ===")
    for split in ("to_P06241", "to_P01053", "to_P0A9X9"):
        s = load(split)
        s["len"] = s["sequence"].str.len()
        tab = pd.crosstab(s["len"], s["set"])
        print(f"\n{split}:\n{tab.to_string()}")


if __name__ == "__main__":
    main()
