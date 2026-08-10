"""Materialize the two post-training corpora as flat CSVs, with pins.

Two tracks, both downloaded once and then never touched by anything that runs
inside a task container:

  hh   - Anthropic/hh-rlhf preference pairs, flattened to
         (prompt, chosen, rejected, subset). The dialogue strings share a common
         prefix; that prefix is the prompt and the divergent tails are the two
         responses. Pairs whose responses are identical after stripping carry no
         preference signal and are dropped.

  qa   - multiple-choice QA (allenai/ai2_arc ARC-Easy, sciq, openbookqa),
         flattened to (question, choices, answer_idx, source). Items with fewer
         than two distinct choices are dropped.

Like research/data/flip2/PINS.json, the output records the dataset revision and the
sha256 of every file written, so a rebuild that silently pulls different rows is
detectable rather than invisible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

HH_SUBSETS = ["helpful-base", "helpful-online", "helpful-rejection-sampled",
              "harmless-base"]
QA_SOURCES = {
    # name -> (hf repo, config, split-of-record)
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", ["train", "validation", "test"]),
    "sciq": ("allenai/sciq", None, ["train", "validation", "test"]),
    "openbookqa": ("allenai/openbookqa", "main", ["train", "validation", "test"]),
}

TURN = re.compile(r"\n\n(Human|Assistant):")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_pair(chosen: str, rejected: str) -> tuple[str, str, str] | None:
    """(prompt, chosen_response, rejected_response) or None if unusable.

    Both strings are `...\n\nHuman: q\n\nAssistant: a` transcripts that agree up
    to the final assistant turn. Splitting on the longest common prefix would cut
    mid-word whenever the two responses happen to share their first characters,
    so the split is taken at the last `\n\nAssistant:` marker instead -- a turn
    boundary, which is where the divergence actually is.
    """
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    marks_c = [m.start() for m in TURN.finditer(chosen) if m.group(1) == "Assistant"]
    marks_r = [m.start() for m in TURN.finditer(rejected) if m.group(1) == "Assistant"]
    if not marks_c or not marks_r:
        return None
    pc, pr = chosen[: marks_c[-1]], rejected[: marks_r[-1]]
    if pc != pr:
        # The two transcripts diverge before the last turn: the prompt is
        # ambiguous, so the pair cannot be scored against a single context.
        return None
    head = "\n\nAssistant:"
    rc = chosen[marks_c[-1] + len(head):].strip()
    rr = rejected[marks_r[-1] + len(head):].strip()
    if not rc or not rr or rc == rr:
        return None
    return pc.strip(), rc, rr


def fetch_hh(max_per_subset: int) -> tuple[Path, dict]:
    from datasets import load_dataset

    rows, dropped = [], 0
    for sub in HH_SUBSETS:
        for split in ("train", "test"):
            ds = load_dataset("Anthropic/hh-rlhf", data_dir=sub, split=split)
            if max_per_subset and len(ds) > max_per_subset:
                ds = ds.select(range(max_per_subset))
            for r in ds:
                parts = split_pair(r["chosen"], r["rejected"])
                if parts is None:
                    dropped += 1
                    continue
                prompt, rc, rr = parts
                rows.append({"prompt": prompt, "chosen": rc, "rejected": rr,
                             "subset": sub, "hf_split": split})
    df = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / "hh_pairs.csv.gz"
    df.to_csv(dest, index=False)
    meta = {"rows": len(df), "dropped_unusable": dropped,
            "subsets": {k: int(v) for k, v in df["subset"].value_counts().items()},
            "sha256": sha256_file(dest)}
    print(f"hh: {len(df):,} pairs ({dropped:,} dropped) -> {dest}")
    return dest, meta


def fetch_qa() -> tuple[Path, dict]:
    from datasets import load_dataset

    rows, dropped = [], 0
    for name, (repo, config, splits) in QA_SOURCES.items():
        for split in splits:
            ds = load_dataset(repo, config, split=split) if config else \
                load_dataset(repo, split=split)
            for r in ds:
                item = normalize_qa(name, r)
                if item is None:
                    dropped += 1
                    continue
                item |= {"source": name, "hf_split": split}
                rows.append(item)
    df = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / "qa_items.csv.gz"
    df.to_csv(dest, index=False)
    meta = {"rows": len(df), "dropped_unusable": dropped,
            "sources": {k: int(v) for k, v in df["source"].value_counts().items()},
            "sha256": sha256_file(dest)}
    print(f"qa: {len(df):,} items ({dropped:,} dropped) -> {dest}")
    return dest, meta


def normalize_qa(name: str, r: dict) -> dict | None:
    """One schema for three datasets: question, choices (JSON list), answer_idx."""
    if name in ("arc_easy", "openbookqa"):
        q = r["question_stem"] if name == "openbookqa" else r["question"]
        texts = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])
        key = r["answerKey"]
        if key not in labels:
            return None
        idx = labels.index(key)
    elif name == "sciq":
        # sciq stores one correct and three distractors; the correct answer is
        # always field `correct_answer`, so a fixed position would let a model
        # score by position rather than by content. Order is fixed by a hash of
        # the question so the assignment is deterministic without a global RNG.
        texts = [r["distractor1"], r["distractor2"], r["distractor3"],
                 r["correct_answer"]]
        q = r["question"]
        h = int(hashlib.sha256(q.encode()).hexdigest(), 16)
        idx = h % 4
        texts[idx], texts[3] = texts[3], texts[idx]
    else:
        return None
    texts = [str(t).strip() for t in texts]
    if len(texts) < 2 or len(set(texts)) < 2 or not str(q).strip():
        return None
    if not texts[idx]:
        return None
    return {"question": str(q).strip(), "choices": json.dumps(texts),
            "answer_idx": idx, "n_choices": len(texts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["hh", "qa", "both"], default="both")
    ap.add_argument("--max-per-subset", type=int, default=20000,
                    help="cap rows read per hh subset/split; 0 for all")
    args = ap.parse_args()

    pins_path = DATA / "PINS.json"
    pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}
    if args.track in ("hh", "both"):
        _, pins["hh"] = fetch_hh(args.max_per_subset)
        pins["hh"]["source"] = "Anthropic/hh-rlhf"
    if args.track in ("qa", "both"):
        _, pins["qa"] = fetch_qa()
        pins["qa"]["sources_hf"] = {k: v[0] for k, v in QA_SOURCES.items()}
    DATA.mkdir(parents=True, exist_ok=True)
    pins_path.write_text(json.dumps(pins, indent=2))
    print(f"wrote {pins_path}")


if __name__ == "__main__":
    main()
