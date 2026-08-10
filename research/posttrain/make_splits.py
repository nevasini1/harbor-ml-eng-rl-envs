"""Build the private splits for both post-training tracks.

Same contract as `research/make_private_split.py` on the mol side:

  split/agent/*     - goes into the agent image, labels included
  split/private/*   - verifier only, never present in the agent container
  split/private/*_fingerprint.json
                    - rare word n-grams that occur in the held-out rows and
                      nowhere in the agent's file, for the grade-time
                      contamination check

The partition is drawn under a seed that is not published (`PRIVATE_SEED`), so
an agent that re-downloads the public corpus cannot reconstruct which rows were
held out -- it can only re-download the rows themselves, which is what the
fingerprint check is for.

Held-out selection is at **prompt** level for preferences and at **item** level
for QA. A preference prompt appears in several pairs with different sampled
responses, so a row-level split would put the same context on both sides and let
a model score by recognising contexts rather than by judging responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "common"))
from textmatch import build_fingerprint  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "split"

# Four candidate eval sets are cut, not two. Which pair ships is decided by the
# measured ladder -- an eval set where a fine-tune does not beat the frozen
# ceiling measures nothing, and the only way to know which ones those are is to
# hold them out and look. The screening run at 1,500 pairs already showed
# `harmless` and `online` sitting at chance for every arm.
RM_EVAL_SUBSETS = {"helpful_base": ["helpful-base"],
                   "helpful_rs": ["helpful-rejection-sampled"],
                   "online": ["helpful-online"],
                   "harmless": ["harmless-base"]}
RM_TRAIN_SUBSETS = ["helpful-base", "helpful-rejection-sampled", "harmless-base"]
QA_SOURCES = ["arc_easy", "sciq", "openbookqa"]


def private_seed() -> int:
    return int((HERE / "PRIVATE_SEED").read_text().strip())


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- preferences

def balance_length(rows: pd.DataFrame, rng) -> pd.DataFrame:
    """Equal numbers of longer-chosen and shorter-chosen pairs; ties dropped.

    Without this the preference track measures response length and nothing else.
    Measured on the unbalanced cut: "pick the longer response" -- a heuristic with
    no parameters and no model -- scored **0.6031** on the `helpful_base` holdout,
    against 0.6042 for a full fine-tune of the pretrained encoder and 0.5646 for
    the frozen-encoder ceiling. On `harmless` the same heuristic scored 0.4116,
    which is why every model arm there landed *below* chance: they learned
    "longer is better" from the training file and it is backwards on that split.
    A randomly-initialized encoder reached 0.594, within seed noise of the
    pretrained fine-tune, so the pretrained weights were doing no work at all.

    Balancing makes the heuristic score exactly 0.5 by construction, which is
    what forces the task back onto content. It is applied to the training file
    as well as the holdouts: leaving the training file biased would teach every
    submission a feature that is worth nothing at grade time, which measures
    whether the agent noticed the trap rather than whether it can post-train.
    """
    lc = rows["chosen"].astype(str).str.len()
    lr = rows["rejected"].astype(str).str.len()
    longer = rows[lc > lr]
    shorter = rows[lc < lr]
    k = min(len(longer), len(shorter))
    out = pd.concat([longer.iloc[rng.permutation(len(longer))[:k]],
                     shorter.iloc[rng.permutation(len(shorter))[:k]]])
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def build_rm(n_train: int, n_test: int, eval_sets: list[str], rng,
             balance: bool = True) -> dict:
    df = pd.read_csv(DATA / "hh_pairs.csv.gz")
    df["prompt"] = df["prompt"].astype(str)
    if balance:
        before = len(df)
        df = balance_length(df, rng)
        print(f"  length-balanced the pool: {before:,} -> {len(df):,} pairs")

    held_prompts: set[str] = set()
    tests: dict[str, pd.DataFrame] = {}
    for name in eval_sets:
        pool = df[df["subset"].isin(RM_EVAL_SUBSETS[name])]
        cand = pd.Index(pool["prompt"].unique()).difference(pd.Index(sorted(held_prompts)))
        # Take only as many prompts as the holdout actually needs. Reserving a
        # fixed multiple of n_test prompts looks harmless and is not: every
        # reserved prompt is removed from the training pool for *all* eval sets,
        # and at four eval sets a 4x multiple left only 16,416 training pairs out
        # of a 66,000-pair corpus. Prompts are accumulated in shuffled order until
        # they carry 1.4x the rows the holdout needs -- enough slack for the
        # balancing trim, and nothing beyond it.
        order = rng.permutation(len(cand))
        counts = pool.groupby("prompt").size()
        picked, have = set(), 0
        for i in order:
            p = cand[i]
            picked.add(p)
            have += int(counts.get(p, 0))
            if have >= n_test * 1.4:
                break
        rows = pool[pool["prompt"].isin(picked)]
        # Balancing the pool leaves a subsample only approximately balanced
        # (binomial, +/-0.8% at n=4,000), so each frame is trimmed exactly. A
        # residual 0.8% edge would be small, but it would be a free 0.008 of
        # accuracy sitting inside a band of 0.04.
        rows = balance_length(rows, rng) if balance else rows
        rows = rows.iloc[rng.permutation(len(rows))[:n_test]].reset_index(drop=True)
        rows = balance_length(rows, rng) if balance else rows
        tests[name] = rows[["prompt", "chosen", "rejected"]]
        held_prompts |= picked

    pool = df[df["subset"].isin(RM_TRAIN_SUBSETS) & ~df["prompt"].isin(held_prompts)]
    train = pool.iloc[rng.permutation(len(pool))[:n_train]].reset_index(drop=True)
    train = balance_length(train, rng) if balance else train
    train = train[["prompt", "chosen", "rejected"]]

    overlap = set(train["prompt"]) & {p for t in tests.values() for p in t["prompt"]}
    if overlap:
        raise SystemExit(f"REFUSING: {len(overlap)} prompts are in both sides")

    (OUT / "agent").mkdir(parents=True, exist_ok=True)
    (OUT / "private").mkdir(parents=True, exist_ok=True)
    train_path = OUT / "agent" / "hh_train.csv.gz"
    train.to_csv(train_path, index=False)

    public_texts = (train["prompt"].tolist() + train["chosen"].tolist() +
                    train["rejected"].tolist())
    manifest = {"n_train": len(train), "eval_sets": {},
                "train_sha256": sha256_file(train_path)}
    fingerprints = {}
    for name, te in tests.items():
        p = OUT / "private" / f"{name}_test.csv"
        te.to_csv(p, index=False)
        private_texts = (te["prompt"].tolist() + te["chosen"].tolist() +
                         te["rejected"].tolist())
        fp = build_fingerprint(private_texts, public_texts)
        fingerprints[name] = fp
        manifest["eval_sets"][name] = {
            "n_test": len(te), "sha256": sha256_file(p), "n_fingerprint": len(fp),
            "subsets": RM_EVAL_SUBSETS[name]}
        print(f"  rm {name}: {len(te):,} pairs, {len(fp):,} private-only shingles")
    (OUT / "private" / "hh_fingerprint.json").write_text(json.dumps(fingerprints))
    return manifest


# ------------------------------------------------------------------------ qa

def build_qa(n_train: int, n_test: int, sources: list[str], rng) -> dict:
    df = pd.read_csv(DATA / "qa_items.csv.gz")
    df = df[df["source"].isin(sources)].reset_index(drop=True)

    held: set[int] = set()
    tests: dict[str, pd.DataFrame] = {}
    for name in sources:
        pool = df.index[df["source"] == name].to_numpy()
        take = pool[rng.permutation(len(pool))[:n_test]]
        tests[name] = df.loc[take, ["question", "choices", "answer_idx"]].reset_index(drop=True)
        held |= set(int(i) for i in take)

    rest = df.drop(index=sorted(held))
    train = rest.iloc[rng.permutation(len(rest))[:n_train]].reset_index(drop=True)
    held_q = {q for t in tests.values() for q in t["question"]}
    before = len(train)
    train = train[~train["question"].isin(held_q)].reset_index(drop=True)
    train = train[["question", "choices", "answer_idx", "source"]]

    (OUT / "agent").mkdir(parents=True, exist_ok=True)
    (OUT / "private").mkdir(parents=True, exist_ok=True)
    train_path = OUT / "agent" / "qa_train.csv"
    train.to_csv(train_path, index=False)

    public_texts = train["question"].tolist() + [
        c for row in train["choices"] for c in json.loads(row)]
    manifest = {"n_train": len(train), "dropped_duplicate_stems": before - len(train),
                "train_sha256": sha256_file(train_path), "eval_sets": {}}
    fingerprints = {}
    for name, te in tests.items():
        p = OUT / "private" / f"{name}_test.csv"
        te.to_csv(p, index=False)
        private_texts = te["question"].tolist() + [
            c for row in te["choices"] for c in json.loads(row)]
        fp = build_fingerprint(private_texts, public_texts)
        fingerprints[name] = fp
        manifest["eval_sets"][name] = {"n_test": len(te), "sha256": sha256_file(p),
                                       "n_fingerprint": len(fp)}
        print(f"  qa {name}: {len(te):,} items, {len(fp):,} private-only shingles")
    (OUT / "private" / "qa_fingerprint.json").write_text(json.dumps(fingerprints))
    return manifest


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["rm", "qa", "both"], default="both")
    ap.add_argument("--rm-train", type=int, default=40000)
    ap.add_argument("--rm-test", type=int, default=1000)
    ap.add_argument("--rm-eval-sets", default="helpful,harmless")
    ap.add_argument("--qa-train", type=int, default=6000)
    ap.add_argument("--qa-test", type=int, default=600)
    ap.add_argument("--qa-sources", default="arc_easy,sciq,openbookqa")
    args = ap.parse_args()

    rng = np.random.default_rng(private_seed())
    manifest_path = OUT / "private" / "manifest.json"
    OUT.mkdir(exist_ok=True)
    (OUT / "private").mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if args.track in ("rm", "both"):
        manifest["rm"] = build_rm(args.rm_train, args.rm_test,
                                  args.rm_eval_sets.split(","), rng)
    if args.track in ("qa", "both"):
        manifest["qa"] = build_qa(args.qa_train, args.qa_test,
                                  args.qa_sources.split(","), rng)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
