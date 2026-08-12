"""Populate the two post-training Harbor task trees from the spike artifacts.

Same shape and same invariant as `research/assemble_task.py` on the mol side:
nothing under `split/private`, no anchor file and no seed may land in an agent
build context. The agent gets training data only; the private rows, the anchors,
the fingerprints and the base-model copy used by the lineage check go exclusively
into the verifier build context.

It also refuses rather than guesses. A results file missing a measured anchor
produces a verifier that fails closed at grade time, which is a broken image
discovered late; catching it here is the same check moved earlier.

    python research/posttrain/assemble_tasks.py --track both
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

sys.path.insert(0, str(ROOT / "common"))
from provenance import stamp  # noqa: E402
from shipping import criterion_record  # noqa: E402

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
# One eval set is ~966s of verifier time for this task; test.sh allows 3,000s.
PROVISIONAL_MAX_EVAL_SETS = 1

# `sigma`, `n_seeds`, `k_screened`, `band_z` and `z_crit_bonferroni` are carried
# because common/shipping.py's report() needs them and, without them, substitutes:
# `n_seeds` defaulted to 5, `k_screened` fell back to a hardcoded table, and `sigma`
# was reconstructed as band/band_sigma with both arms assumed equally noisy. The mol
# assembler carries them, so one printed table mixed mol's recorded z-values with
# reconstructed ones for these two tracks, in the same column, unmarked. report()'s
# own docstring calls printing a reconstruction beside a file that states the real
# number "two answers to one question -- the thing this criterion exists to stop".
#
# `gate_a` is carried so an eval set that shipped without a random-init control says
# so in the file rather than only in the report.
CARRY = ("base_acc", "reference_acc", "t_implausible", "base_arm",
         "base_definition", "reference_definition", "band", "band_sigma",
         "sigma", "n_seeds", "k_screened", "band_z", "z_crit_bonferroni",
         "reward_noise_on_rerun", "min_band_sigma", "pretraining_gain", "gate_a",
         "best_observed", "provisional", "provisional_reason")


def download(repo: str, revision: str, dest: Path, patterns: list[str]) -> None:
    if dest.is_dir() and (dest / "config.json").exists():
        print(f"    have {dest.relative_to(ROOT)}")
        return
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, revision=revision, local_dir=str(dest),
                      allow_patterns=patterns)
    print(f"    fetched {repo}@{revision[:12]} -> {dest.relative_to(ROOT)}")


def assemble(track: str, cfg: dict, allow_provisional: bool = False) -> None:
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
    doc = json.loads(measured_path.read_text())
    measured = doc["anchors"]
    if not measured:
        # Every eval set failed common/shipping.py. The task is a complete
        # environment and its recorded runs are real, so it can still be
        # assembled -- but only on purpose, and only stamped as such.
        if not allow_provisional:
            reasons = "; ".join(f"{k}: {v}" for k, v in doc.get("rejected", {}).items())
            raise SystemExit(
                f"REFUSING: no eval set in {cfg['anchors']} passes the shipping "
                f"criterion.\n  {reasons}\n"
                "Pass --allow-provisional to assemble it anyway; the anchors will be "
                "stamped provisional and the reward must not be treated as validated.")
        # Cap at the single best screened set, by band_sigma. Not a stylistic
        # choice: this verifier was measured at 1,933s for two eval sets against
        # test.sh's 3,000s allowance, so four would be killed mid-run and score 0.
        # The cap is announced rather than applied silently.
        ranked = sorted(((k, v) for k, v in doc["screened"].items()
                         if k in doc.get("rejected", {})),
                        key=lambda kv: kv[1].get("band_sigma", 0), reverse=True)
        keep, dropped = ranked[:PROVISIONAL_MAX_EVAL_SETS], \
            ranked[PROVISIONAL_MAX_EVAL_SETS:]
        measured = {k: dict(v, provisional=True,
                            provisional_reason=doc["rejected"][k]) for k, v in keep}
        print(f"    PROVISIONAL: no eval set passes the shipping criterion; "
              f"assembling the best {len(measured)} of {len(ranked)} screened "
              f"(kept {[k for k, _ in keep]}, dropped {[k for k, _ in dropped]} "
              f"— verifier wall time, not quality)")

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
    # Sidecar metadata, separated from the eval sets by verifier_core.load_anchors.
    # It lives in this file rather than beside it so the rule an anchor was screened
    # by, and the commit that wrote it, cannot drift away from the anchor.
    c = criterion_record()
    anchors["_criterion"] = c
    anchors["_provenance"] = stamp(
        "research/posttrain/assemble_tasks.py",
        source=f"research/posttrain/results/{cfg['anchors']}",
        measured_by="research/posttrain/modal_measure.py",
        derived_by="research/posttrain/finalize_anchors.py",
        note="assembled_at is when these anchors were written into the task tree, "
             "not when the seed runs happened; that date predates this field and is "
             "not recoverable, so it is not claimed here.")
    (priv / "anchors.json").write_text(json.dumps(anchors, indent=2))
    print(f"    criterion: {c['rule_id']} v{c['rule_version']}, "
          f"{c['min_band_sigma']} sigma")
    g = anchors["_provenance"]["git"]
    print(f"    provenance: {g.get('commit', 'no git')}"
          + (" (DIRTY TREE)" if g.get("dirty") else ""))

    # Prune held-out CSVs for eval sets that are no longer shipped. Without this,
    # tightening the criterion leaves the previous run's private rows inside the
    # verifier build context: dead weight in the image, and held-out data the
    # grader no longer accounts for.
    for stale in priv.glob("*_test.csv"):
        if stale.stem[:-len("_test")] not in anchors:
            stale.unlink()
            print(f"    pruned stale private split: {stale.name}")

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
    ap.add_argument("--allow-provisional", action="store_true",
                    help="assemble a task whose eval sets all fail common/shipping.py")
    args = ap.parse_args()

    for track, cfg in TRACKS.items():
        if args.track in (track, "both"):
            assemble(track, cfg, args.allow_provisional)

    # Keep the shared grader modules in step with common/; a task whose copy has
    # drifted grades with code the repo no longer shows.
    print()
    subprocess.run([sys.executable, str(ROOT / "common" / "sync.py")], check=True)


if __name__ == "__main__":
    main()
