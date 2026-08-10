"""Measure both post-training ladders on GPUs, one container per (arm, seed).

Why remote: every number that ends up in an `anchors.json` is a mean over seeds,
and a seeded fine-tune of `distilroberta-base` takes ~20 minutes per seed on the
8 CPU cores available locally. Five seeds x four arms x two tracks is more than a
day of local wall time, which is how anchors end up being *chosen* instead of
measured. On an A10G each arm is minutes, and the grid runs in parallel.

Why this does not change what is being measured: the reward is a function of the
submitted weights, evaluated on CPU inside the verifier. Where those weights were
produced does not enter it. What *is* CPU-specific -- whether the reference recipe
fits the agent's 4-hour budget -- is measured separately, on the same 8 cores the
agent gets, and recorded in RESULTS.md.

    modal run research/posttrain/modal_measure.py --track rm
    modal run research/posttrain/modal_measure.py --track qa

Results land in research/posttrain/results/{rm,qa}_anchors.json, which is what
assemble_tasks.py reads. That file is refused by the assembler if an anchor is
missing, so a partial grid cannot quietly become a shipped task.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
REMOTE = "/work"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "safetensors==0.4.5",
        "scikit-learn==1.5.2",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "huggingface_hub==0.28.1",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(f"{HERE}/rm_ladder.py", f"{REMOTE}/rm_ladder.py")
    .add_local_file(f"{HERE}/sft_ladder.py", f"{REMOTE}/sft_ladder.py")
    .add_local_dir(f"{HERE}/split", f"{REMOTE}/split")
)

cache = modal.Volume.from_name("posttrain-hf-cache", create_if_missing=True)
app = modal.App("posttrain-ladders")


@app.function(image=image, gpu="A10G", timeout=60 * 60,
              volumes={"/cache": cache}, max_containers=12)
def run_arm(track: str, arm: str, seed: int, cfg: dict) -> dict:
    """One arm at one seed. Returns {eval_set: metric} plus `_`-prefixed detail."""
    import sys
    import time

    sys.path.insert(0, REMOTE)
    import pandas as pd
    import torch

    split = Path(REMOTE) / "split"
    eval_sets = cfg["eval_sets"]
    t0 = time.time()

    if track == "rm":
        import rm_ladder as L

        L.BASE_MODEL, L.MAX_LEN = cfg["base_model"], cfg["max_len"]
        train = pd.read_csv(split / "agent" / "hh_train.csv.gz")
        if cfg.get("n_train"):
            train = train.head(cfg["n_train"])
        tests = {n: pd.read_csv(split / "private" / f"{n}_test.csv") for n in eval_sets}

        if arm == "length_only":
            out = L.arm_length_only(tests)
        elif arm in ("frozen_probe", "frozen_head"):
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(L.BASE_MODEL)
            enc = AutoModel.from_pretrained(L.BASE_MODEL).to("cuda")
            feats = {"train": L.embed_frame(enc, tok, train, "cuda"),
                     "tests": {k: L.embed_frame(enc, tok, v, "cuda")
                               for k, v in tests.items()}}
            del enc
            out = (L.arm_frozen_probe(feats, seed) if arm == "frozen_probe"
                   else L.arm_frozen_head(feats, seed))
        else:
            out = L.arm_finetune(train, tests, seed, "cuda", cfg["epochs"],
                                 cfg["lr"], cfg["head_lr"], cfg["bs"],
                                 random_init=(arm == "random_init"))
    else:
        import sft_ladder as L

        L.BASE_MODEL, L.MAX_LEN = cfg["base_model"], cfg["max_len"]
        train = pd.read_csv(split / "agent" / "qa_train.csv")
        if cfg.get("n_train"):
            train = train.head(cfg["n_train"])
        tests = {n: pd.read_csv(split / "private" / f"{n}_test.csv") for n in eval_sets}

        if arm == "zero_shot":
            out = L.zero_shot(tests, "cuda")
        else:
            mode = {"sft_full": "full", "head_only": "head",
                    "random_init": "random_init"}[arm]
            lr = 1e-3 if mode == "head" else cfg["lr"]
            out = L.sft(train, tests, seed, "cuda", cfg["epochs"], lr, cfg["bs"], mode)

    out["_seconds"] = round(time.time() - t0, 1)
    out["_gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"{track}/{arm}/seed{seed}: " +
          "  ".join(f"{k}={v:.4f}" for k, v in out.items()
                    if not k.startswith("_")), flush=True)
    return {"arm": arm, "seed": seed, **out}


TRACK_DEFAULTS = {
    "rm": {"base_model": "distilroberta-base", "max_len": 256, "epochs": 2,
           "lr": 2e-5, "head_lr": 1e-3, "bs": 8, "n_train": 0,
           "eval_sets": ["helpful_base", "helpful_rs", "online", "harmless"],
           "arms": ["length_only", "frozen_probe", "frozen_head", "finetune",
                    "random_init"],
           "metric": "acc"},
    "qa": {"base_model": "HuggingFaceTB/SmolLM2-135M", "max_len": 160, "epochs": 3,
           "lr": 2e-5, "bs": 16, "n_train": 0,
           "eval_sets": ["arc_easy", "sciq", "openbookqa"],
           "arms": ["zero_shot", "head_only", "sft_full", "random_init"],
           "metric": "acc"},
}


def summarize(track: str, cfg: dict, rows: list[dict]) -> dict:
    """Per-arm mean/std over seeds. This file measures; it does not decide.

    Turning the ladder into anchors -- which arm is the ceiling, what the tripwire
    is, whether an eval set has enough band to ship -- lives in
    `finalize_anchors.py`, so the rules can be re-read and re-run without a GPU.
    """
    import statistics as st

    arms: dict[str, dict] = {}
    for arm in cfg["arms"]:
        vals = {n: [r[n] for r in rows if r["arm"] == arm] for n in cfg["eval_sets"]}
        arms[arm] = {n: {"mean": round(st.fmean(v), 4),
                         "std": round(st.stdev(v), 4) if len(v) > 1 else 0.0,
                         "seeds": [round(x, 4) for x in v]}
                     for n, v in vals.items() if v}

    return {"config": cfg, "arms": arms}


@app.local_entrypoint()
def main(track: str = "rm", seeds: int = 5, arms: str = "", epochs: int = 0,
         n_train: int = 0, out: str = ""):
    cfg = dict(TRACK_DEFAULTS[track])
    if arms:
        cfg["arms"] = arms.split(",")
    if epochs:
        cfg["epochs"] = epochs
    if n_train:
        cfg["n_train"] = n_train

    grid = [(track, arm, s, cfg)
            for arm in cfg["arms"]
            # zero_shot and length_only are deterministic; extra seeds would be
            # identical runs.
            for s in (range(1) if arm in ("zero_shot", "length_only")
                      else range(seeds))]
    print(f"{track}: {len(grid)} containers "
          f"({len(cfg['arms'])} arms x up to {seeds} seeds)")

    rows = list(run_arm.starmap(grid))
    summary = summarize(track, cfg, rows)

    # Relative --out is resolved against this file, not the shell's cwd. A run
    # launched from the repo root was writing results/ at the root instead of
    # into research/posttrain/results/, which is the directory every other script
    # reads.
    dest = (Path(out) if Path(out).is_absolute() else HERE / out) if out \
        else HERE / "results" / f"{track}_anchors.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2))

    names = cfg["eval_sets"]
    print(f"\n{'arm':<14}" + "".join(f"{n:>14}" for n in names))
    for arm, agg in summary["arms"].items():
        print(f"{arm:<14}" + "".join(
            f"{agg[n]['mean']:>9.4f}±{agg[n]['std']:.3f}" if n in agg else f"{'-':>14}"
            for n in names))
    print(f"\nwrote {dest}")
    print(f"next: python research/posttrain/finalize_anchors.py --track {track}")
