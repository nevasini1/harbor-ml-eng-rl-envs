"""Inventory the downloaded FLIP2 splits: schema, row counts, sequence lengths."""

import gzip
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "flip2"


def summarize(path: Path) -> None:
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh)

    rel = path.relative_to(DATA)
    lens = df["sequence"].str.len() if "sequence" in df.columns else None

    print(f"\n=== {rel} ===")
    print(f"  columns : {list(df.columns)}")
    print(f"  rows    : {len(df):,}")
    if lens is not None:
        print(f"  seq len : min={lens.min()} median={int(lens.median())} max={lens.max()}")
        print(f"  n_unique_seq: {df['sequence'].nunique():,}")
    for col in ("set", "validation"):
        if col in df.columns:
            counts = df[col].value_counts(dropna=False).to_dict()
            print(f"  {col:<11}: {counts}")
    if "target" in df.columns:
        t = pd.to_numeric(df["target"], errors="coerce")
        print(
            f"  target  : min={t.min():.3f} median={t.median():.3f} "
            f"max={t.max():.3f} n_nan={t.isna().sum()}"
        )
    extra = [c for c in df.columns if c not in ("sequence", "target", "set", "validation")]
    for col in extra:
        nu = df[col].nunique()
        head = df[col].dropna().unique()[:4]
        print(f"  {col:<11}: {nu} unique, e.g. {list(head)}")


def main() -> None:
    for path in sorted(DATA.rglob("*.csv.gz")):
        summarize(path)


if __name__ == "__main__":
    main()
