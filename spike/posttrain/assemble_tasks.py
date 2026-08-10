"""Populate the two post-training Harbor task trees from the spike artifacts.

Same shape and same invariant as `spike/assemble_task.py` on the mol side:
nothing under `split/private`, no anchor file and no seed may land in an agent
build context. The agent gets training data only; the private rows, the anchors,
the fingerprints and the base-model copy used by the lineage check go exclusively
into the verifier build context.

It also refuses rather than guesses. A results file missing a measured anchor
produces a verifier that fails closed at grade time, which is a broken image
discovered late; catching it here is the same check moved earlier.

    python spike/posttrain/assemble_tasks.py --track both
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SPLIT = HERE / "split"
RESULTS = HERE / "results"

TRACKS = {
    "rm": {
        "task": ROOT / "tasks" / "pref-reward-model",
        "train_file": "hh_train.csv.gz",
        "fingerprint": "hh_fingerprint.json",
        "anchors": "rm_anchors.json",
        "metric": "acc",
        "base_repo": "distilroberta-base",
        "base_revision": "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b",
        "base_patterns": ["config.json", "pytorch_model.bin", "vocab.json",
                          "merges.txt", "tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json"],
        "siblings": {},
    },
    "qa": {
        "task": ROOT / "tasks" / "qa-sft-adapt",
        "train_file": "qa_train.csv",
        "fingerprint": "qa_fingerprint.json",
        "anchors": "qa_anchors.json",
        "metric": "acc",
        "base_repo": "HuggingFaceTB/SmolLM2-135M",
        "base_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "base_patterns": ["config.json", "model.safetensors", "tokenizer.json",
                          "tokenizer_config.json", "special_tokens_map.json",
                          "vocab.json", "merges.txt"],
        # Same-architecture public checkpoint, baked in for check_closer_to_base.
        "siblings": {
            "SmolLM2-135M-Instruct": (
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
        },
    },
}

# Carried through from the measurement, not restated here. The mol assembler had
# these hardcoded once and they drifted: the shipped definitions still described
# a pooling the anchors had not used for weeks.
CARRY = ("base_acc", "reference_acc", "t_implausible", "base_arm",
         "base_definition", "reference_definition", "band", "band_sigma")


def download(repo: str, revision: str, dest: Path, patterns: list[str]) -> None:
    if dest.is_dir() and (dest / "config.json").exists():
        print(f"    have {dest.relative_to(ROOT)}")
        return
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, revision=revision, local_dir=str(dest),
                      allow_patterns=patterns)
    print(f"    fetched {repo}@{revision[:12]} -> {dest.relative_to(ROOT)}")


def assemble(track: str, cfg: dict) -> None:
    task = cfg["task"]
    agent_data = task / "environment" / "data"
    grader = task / "tests" / "grader"
    priv = grader / "private"
    agent_data.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)

    print(f"\n== {task.name}")
    shutil.copy2(SPLIT / "agent" / cfg["train_file"], agent_data / cfg["train_file"])
    print(f"    agent data: {cfg['train_file']}")

    measured_path = RESULTS / cfg["anchors"]
    if not measured_path.exists():
        raise SystemExit(
            f"REFUSING: {measured_path} does not exist. The anchors are a "
            "measurement; there is no default to fall back on.")
    measured = json.loads(measured_path.read_text())["anchors"]

    anchors = {}
    for name, m in measured.items():
        missing = [k for k in ("base_acc", "reference_acc", "t_implausible")
                   if k not in m]
        if missing:
            raise SystemExit(
                f"REFUSING: {cfg['anchors']}[{name}] is missing {missing}. "
                "grade.py fails closed on these, so a partial anchor file would "
                "produce a verifier that cannot run.")
        anchors[name] = {k: m[k] for k in CARRY if k in m}
        shutil.copy2(SPLIT / "private" / f"{name}_test.csv", priv / f"{name}_test.csv")
        print(f"    private: {name}_test.csv  base={m['base_acc']} "
              f"reference={m['reference_acc']}")
    (priv / "anchors.json").write_text(json.dumps(anchors, indent=2))

    fp_src = SPLIT / "private" / cfg["fingerprint"]
    fp = json.loads(fp_src.read_text())
    unused = set(fp) - set(anchors)
    if set(anchors) - set(fp):
        raise SystemExit(
            f"REFUSING: no fingerprint for {sorted(set(anchors) - set(fp))}; "
            "the contamination check would silently skip those eval sets.")
    # Eval sets that were screened but not shipped keep their rows out of the
    # image, so their fingerprints would only bloat it.
    (priv / cfg["fingerprint"]).write_text(
        json.dumps({k: v for k, v in fp.items() if k in anchors}))
    print(f"    fingerprint: {len(anchors)} eval sets"
          + (f" ({len(unused)} screened-but-unshipped dropped)" if unused else ""))

    shutil.copy2(RESULTS / "public_hashes.json", grader / "public_hashes.json")
    download(cfg["base_repo"], cfg["base_revision"], grader / "base_model",
             cfg["base_patterns"])
    for name, (repo, rev) in cfg["siblings"].items():
        download(repo, rev, grader / "siblings" / name, cfg["base_patterns"])

    # The lineage check reads /grader/base_model; a missing fixture means the
    # image fails to build, or worse, builds and rejects every honest run.
    if not (grader / "base_model" / "config.json").exists():
        raise SystemExit(f"REFUSING: no base-model fixture under {grader}")

    leaked = [p.name for p in agent_data.iterdir()
              if p.name != cfg["train_file"]]
    if leaked:
        raise SystemExit(f"REFUSING: unexpected files in the agent image: {leaked}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["rm", "qa", "both"], default="both")
    args = ap.parse_args()

    for track, cfg in TRACKS.items():
        if args.track in (track, "both"):
            assemble(track, cfg)

    # Keep the shared grader modules in step with common/; a task whose copy has
    # drifted grades with code the repo no longer shows.
    print()
    subprocess.run([sys.executable, str(ROOT / "common" / "sync.py")], check=True)


if __name__ == "__main__":
    main()
