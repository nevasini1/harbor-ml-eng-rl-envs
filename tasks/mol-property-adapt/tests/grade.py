"""Verifier for mleval/mol-property-adapt.

Contract: the agent submits /app/final_model/<eval_set>/ as a `save_pretrained`
directory loadable by AutoModelForSequenceClassification. No agent-authored code
is imported or executed here.

reward = mean over eval sets of clip((auc - base) / (reference - base), 0, 1)

The integrity layers, the fail-closed anchor loader and the always-write driver
now live in `verifier_core.py`, shared with the other tasks in this repo; this
file is what is specific to molecules. The extraction is behaviour-preserving:
regrading `jobs/mol-oracle-modal` still returns 0.909661.

Design constraints the shared core is written against (restated because they are
what makes this grader trustworthy):

  * Never crash into ambiguity. Any agent output - missing, malformed, hostile,
    wrong shape, non-finite, or a model that raises at inference - must still
    produce a reward. Every failure path floors that eval set to 0.0 and the run
    continues; the process always writes reward.json.

  * Three anti-substitution layers, because they fail on disjoint inputs:
    architecture-config hash, sha256 vs known public checkpoints, and per-tensor
    float64 cosine vs the base encoder. The third deliberately ALLOWS an
    unmodified encoder: freezing the backbone and training only a head is a
    legitimate strategy, so "weights must have moved" is not a valid requirement.

  * reward.json carries exactly one key. Harbor's default dataset metric raises
    on a multi-key reward dict, so per-eval-set detail goes to metrics.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from pathlib import Path

from verifier_core import (Reject, check_architecture, check_lineage,
                           check_not_other_public, grade_eval_sets, load_anchors,
                           load_public_hashes, scan_dir)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MAX_SMILES_CHARS = 1024
BATCH_SIZE = 64

# The base MLM exposes 53 encoder tensors once heads are excluded, so a genuine
# derivative compares 53. The old floor of 10 was far too loose: the comparison
# loop skips every tensor that does not match the base by name and shape, so an
# unrelated checkpoint carrying a handful of copied embedding tensors cleared it.
MIN_ENCODER_TENSORS = 50

# The AUC above which a score indicates test-set contamination rather than skill
# is a tripwire for a human to review, not proof. It is no longer stated here: the
# rule is `min(0.98, max(best_observed + 0.15, 0.85))` in research/assemble_task.py,
# the same one research/posttrain/finalize_anchors.py applies to the other tracks,
# and the derived value travels in anchors.json. This comment used to justify a
# hardcoded 0.85 by citing the anchors and the best of 25 seeded runs (0.7111) --
# a justification that would have gone stale the moment either was re-measured,
# which they since were.
#
# No default for the contamination tripwire. Every anchor this grader is built
# with carries `t_implausible` -- the post-training tracks derive it in
# finalize_anchors.py and the mol assembler derives it in assemble_task.py, both by
# the same stated rule -- so a missing key means a malformed anchors.json, not a
# case to paper over. Passing None makes load_anchors raise with
# "the contamination tripwire would silently not run" instead of scoring the whole
# field against a literal nobody chose in this file.
DEFAULT_T_IMPLAUSIBLE = None


# ------------------------------------------------- test-set contamination

# Artifacts the agent ships that could carry molecules. task.toml copies exactly
# these two paths across, so this is the whole reachable surface.
#
# The agent log is scanned FIRST, the same ordering the two post-training graders
# carry with the note "Order is the whole fix". It matters more here, not less:
# MAX_TOKENS_SCANNED is 500,000 against textmatch's 4,000,000, `.json` is in
# SCANNABLE_SUFFIXES, and this task ships TWO submission directories -- so
# scanning `final_model` first spends the whole budget on two copies of
# `tokenizer.json`, `vocab.json` and `merges.txt` before reaching `train_log.txt`,
# which is where a leak is actually likely to be. Both roots are still scanned.
ARTIFACT_ROOTS = ("/logs/agent", "/app/final_model")
SCANNABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".smi", ".md"}
MAX_SCAN_BYTES = 64 << 20
MAX_TOKENS_SCANNED = 500_000

_ARTIFACT_KEYS: set[str] | None = None


def artifact_inchikeys() -> set[str]:
    """InChIKeys of every molecule parseable out of the agent's artifacts.

    Matching on InChIKey rather than the SMILES string is the point: the same
    molecule has many valid SMILES spellings, so a string comparison would miss
    anything rewritten, canonicalised by a different toolkit, or round-tripped
    through another format. The InChIKey is invariant to all of that.

    Only text-shaped files are read; weights are skipped. Computed once per run
    because the artifacts are shared across eval sets.
    """
    global _ARTIFACT_KEYS
    if _ARTIFACT_KEYS is not None:
        return _ARTIFACT_KEYS

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    seen: set[str] = set()
    budget = MAX_TOKENS_SCANNED
    for root in ARTIFACT_ROOTS:
        base = Path(root)
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if budget <= 0:
                break
            if p.is_symlink() or not p.is_file():
                continue
            # Look THROUGH a .gz suffix, not past it. `p.suffix` for `leak.csv.gz`
            # is ".gz", which is in no suffix set, so a gzipped molecule dump used
            # to skip this scan entirely -- and gzip is the format this repo hands
            # data out in. The `suffixes[-2:]` join above looks like it covers
            # ".csv.gz" but that string is not in SCANNABLE_SUFFIXES either, so it
            # never matched.
            gzipped = p.suffix.lower() == ".gz"
            inner = Path(p.stem).suffix.lower() if gzipped else p.suffix.lower()
            if inner not in SCANNABLE_SUFFIXES:
                continue
            try:
                if p.stat().st_size > MAX_SCAN_BYTES:
                    continue
                if gzipped:
                    with gzip.open(p, "rt", errors="ignore") as fh:
                        text = fh.read(MAX_SCAN_BYTES)
                else:
                    text = p.read_text(errors="ignore")
            except Exception:
                continue
            for tok in re.split(r"[\s,;\"'\[\]{}]+", text):
                if budget <= 0:
                    break
                if not 4 <= len(tok) <= MAX_SMILES_CHARS:
                    continue
                # Cheap pre-filter: real SMILES carry ring digits, branches,
                # bonds or brackets, or are short all-alpha formulas.
                if not (any(c in tok for c in "()=#@+-\\/") or
                        any(c.isdigit() for c in tok)):
                    if not tok.isalpha():
                        continue
                budget -= 1
                try:
                    m = Chem.MolFromSmiles(tok)
                except Exception:
                    continue
                if m is not None and m.GetNumAtoms() > 0:
                    try:
                        seen.add(Chem.MolToInchiKey(m))
                    except Exception:
                        continue
    _ARTIFACT_KEYS = seen
    return seen


def load_private_keys(path: Path) -> dict:
    """Held-out InChIKeys per eval set. Fails closed, like the anchors.

    instruction.md rule 1 promises the agent that "the verifier checks for
    train/test contamination". If this file is missing the check silently does
    not run while the promise still stands, which is precisely the failure mode
    the anchors loader was hardened against.
    """
    if not path.exists():
        raise RuntimeError(
            f"test_inchikeys.json missing at {path}; the contamination check "
            "instruction.md promises cannot run and the verifier image is broken")
    keys = json.loads(path.read_text())
    if not keys:
        raise RuntimeError(f"test_inchikeys.json at {path} is empty")
    return {k: set(v) for k, v in keys.items()}


def check_contamination(name: str, private_keys: set[str]) -> dict:
    """Reject if the held-out molecules turn up in what the agent shipped.

    Direct evidence, unlike the t_implausible tripwire, which only infers
    contamination from an improbable score and so cannot see partial leakage.
    Neither is sufficient alone: this one sees nothing if the agent leaves no
    data behind, and the tripwire sees nothing below its threshold.

    Any overlap is a violation. The private test molecules are absent from the
    agent's training file by construction (train_test_inchikey_overlap is 0 in
    the manifest), so their presence means they came from outside the container.
    """
    found = artifact_inchikeys()
    overlap = private_keys & found
    info = {"artifact_molecules_seen": len(found),
            "private_test_overlap": len(overlap)}
    if overlap:
        raise Reject(
            f"{len(overlap)} held-out {name} molecules found in the submitted "
            f"artifacts; the private split is not derivable from the agent's data")
    return info


# ------------------------------------------------------------------ scoring

def mean_auc(y_true, y_score) -> tuple[float, int]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    aucs = []
    for t in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, t])
        if m.sum() == 0 or len(np.unique(y_true[m, t])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[m, t], y_score[m, t]))
    if not aucs:
        raise Reject("no scoreable tasks in the held-out labels")
    return float(np.mean(aucs)), len(aucs)


def integrity_checks(sub: Path, base: Path, public: dict, base_repo: str) -> dict:
    """The three anti-substitution layers for one submitted model directory.

    Split out of `score_eval_set` so it can run as a pre-pass over EVERY eval set
    before any of them is scored. This task is the only one that accepts a separate
    model per eval set (`/app/final_model/<eval_set>/`), and it used to check each
    one only as that eval set was scored -- so substituting a public checkpoint for
    tox21 alone got tox21 rejected at 0 and bbbp scored normally, for a reward of
    mean(0.0, 1.0) = 0.5 rather than 0.

    The other two anchor-scored tasks cache a single model, so any integrity failure
    floors every eval set and the reward really is 0. README.md's
    `reward = integrity_gate x mean(recovery)` described their shape, not this one.
    Partial credit for a partially substituted submission is not defensible, so the
    gate is now global here too: one failure rejects the whole submission.
    """
    info = {"submission_bytes": scan_dir(sub)}
    info |= check_architecture(sub, base)
    info |= check_not_other_public(sub, public, base_repo)
    info |= check_lineage(sub, base, MIN_ENCODER_TENSORS)
    return info


def score_eval_set(sub: Path, base: Path, test_csv: Path, public: dict,
                   base_repo: str, n_tasks: int, info: dict | None = None) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.use_deterministic_algorithms(True)

    # Pre-computed by the global gate in main(). Recomputed here only if this
    # function is called directly, which the regrade helpers do.
    info = dict(info) if info is not None else integrity_checks(
        sub, base, public, base_repo)

    df = pd.read_csv(test_csv)
    labels = [c for c in df.columns if c != "smiles"]
    if len(labels) != n_tasks:
        raise Reject(f"internal: expected {n_tasks} label columns, found {len(labels)}")
    smiles = [s[:MAX_SMILES_CHARS] for s in df["smiles"].astype(str)]
    Y = df[labels].to_numpy(dtype=np.float64)

    tok = AutoTokenizer.from_pretrained(str(sub))
    model = AutoModelForSequenceClassification.from_pretrained(str(sub))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(smiles), BATCH_SIZE):
            enc = tok(smiles[i : i + BATCH_SIZE], return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            out = model(**enc).logits
            if out.ndim != 2 or out.shape[1] != n_tasks:
                raise Reject(f"expected logits of shape (n,{n_tasks}), "
                             f"got {tuple(out.shape)}")
            preds.append(out.float().numpy())

    P = np.concatenate(preds)
    if not np.all(np.isfinite(P)):
        raise Reject("model produced non-finite outputs")

    auc, n_scored = mean_auc(Y, P)
    info |= {"auc": round(auc, 6), "n_test": len(df), "n_tasks_scored": n_scored}
    return info


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/app/final_model")
    ap.add_argument("--base", default="/grader/base_model")
    ap.add_argument("--private", default="/grader/private")
    ap.add_argument("--anchors", default="/grader/private/anchors.json")
    ap.add_argument("--test-keys", default="/grader/private/test_inchikeys.json")
    ap.add_argument("--public-hashes", default="/grader/public_hashes.json")
    ap.add_argument("--base-repo", default="DeepChem/ChemBERTa-77M-MLM")
    ap.add_argument("--out", default="/logs/verifier/reward.json")
    ap.add_argument("--metrics-out", default="/logs/verifier/metrics.json")
    args = ap.parse_args()

    out, metrics_out = Path(args.out), Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        anchors = load_anchors(Path(args.anchors), "auc", ("n_tasks",),
                               DEFAULT_T_IMPLAUSIBLE)
        private_keys = load_private_keys(Path(args.test_keys))
        missing = set(anchors) - set(private_keys)
        if missing:
            raise RuntimeError(
                f"no held-out InChIKeys for {sorted(missing)}; the contamination "
                "check would silently skip those eval sets")
        public = load_public_hashes(Path(args.public_hashes))
    except BaseException as exc:  # noqa: BLE001
        # Config failures are not the agent's doing, so they are reported as a
        # grader error rather than as a rejected submission - but they still
        # write a reward, because a missing reward.json is a trial error.
        import traceback

        out.write_text(json.dumps({"reward": 0.0}))
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(
            {"eval_sets": {}, "status": "grader_error", "reward": 0.0,
             "reason": f"{type(exc).__name__}: {exc}",
             "traceback": traceback.format_exc()[-1500:]}, indent=2))
        print(f"grader_error: {type(exc).__name__}: {exc}")
        return 0

    # Global integrity gate. Every submitted model is checked BEFORE any eval set is
    # scored, and one failure rejects all of them -- otherwise substituting a single
    # eval set's checkpoint earns full credit on the others. The reason names the
    # eval set that failed, so the zero stays attributable.
    integrity: dict[str, dict] = {}
    gate_failure: str | None = None
    for name in anchors:
        try:
            integrity[name] = integrity_checks(
                Path(args.submission) / name, Path(args.base), public,
                args.base_repo)
        except Reject as exc:
            gate_failure = f"integrity gate failed on eval set {name}: {exc}"
            break
    if gate_failure:
        print(f"integrity gate: REJECT -- {gate_failure}")

    def score_one(name: str, anc: dict) -> dict:
        if gate_failure:
            raise Reject(gate_failure)
        return score_eval_set(
            Path(args.submission) / name, Path(args.base),
            Path(args.private) / f"{name}_test.csv", public, args.base_repo,
            anc["n_tasks"], info=integrity[name])

    grade_eval_sets(
        anchors,
        score_one,
        metric="auc",
        out=out,
        metrics_out=metrics_out,
        default_t_implausible=DEFAULT_T_IMPLAUSIBLE,
        postcheck=lambda name, anc, res: check_contamination(name, private_keys[name]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
