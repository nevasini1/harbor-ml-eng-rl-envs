"""Verifier for mleval/pref-reward-model.

Contract: the agent submits **one** `/app/final_model` -- a `save_pretrained`
directory loadable by AutoModelForSequenceClassification with `num_labels = 1`,
plus its tokenizer. The same model is scored on every eval set. No agent-authored
code is imported or executed here.

reward = mean over eval sets of clip((acc - base) / (reference - base), 0, 1)

where `acc` is pairwise accuracy: the fraction of held-out preference pairs whose
human-preferred response the model scores above the other. Chance is exactly 0.5,
which is what makes the band interpretable without further calibration.

Everything that is not specific to preferences -- the integrity layers, the
fail-closed anchor loader, the always-write driver -- lives in `verifier_core.py`
and is shared with the mol and QA tasks. What is here:

  * `render`, the single input format. The verifier scores a bare checkpoint, so
    if two submissions expected different renderings their scores would not be
    comparable and the anchors would mean nothing. The format is stated in
    instruction.md and applied here; the agent's own tokenizer is used, but the
    truncation side is forced, because a tokenizer saved with the default right
    truncation would silently drop the response being judged.

  * shingle-overlap contamination, the text analogue of the mol task's InChIKey
    check. See common/textmatch.py.

  * ties count as half. A model that emits a constant score is wrong, not
    correct, and counting ties as wins would let it take chance-level accuracy
    for free; counting them as losses would push it below chance and make the
    metric non-monotone in "how much signal does this model have". Half is the
    only choice that leaves an uninformative model at exactly chance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from textmatch import artifact_shingles
from verifier_core import (Reject, check_architecture, check_lineage,
                           check_not_other_public, count_body_tensors,
                           grade_eval_sets, load_anchors, load_public_hashes,
                           scan_dir)

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MAX_LEN = 256
BATCH_SIZE = 32

# Floor for the lineage check, as a fraction of the tensors a perfect derivative
# would expose rather than as a constant. The mol task hardcoded 50 against a
# 53-tensor base, which is correct for that checkpoint and silently wrong for any
# other; a ratio survives a change of base.
MIN_TENSOR_RATIO = 0.9

# Pairwise accuracy above which a score indicates the held-out pairs were seen
# rather than skill. Published reward-model accuracies on hh-rlhf sit near
# 0.65-0.72 for models 10-100x this size, the measured reference here is 0.63,
# and a model trained on the recovered test rows scores >0.95. The tripwire the
# anchors carry sits far above anything an honest run at this scale has produced --
# it is derived in finalize_anchors.py rather than stated here, so re-measuring the
# ceiling moves it. A tripwire for a human to review, not proof.
#
# No default for the contamination tripwire. Every anchor this grader is built
# with carries `t_implausible` -- the post-training tracks derive it in
# finalize_anchors.py and the mol assembler derives it in assemble_task.py, both by
# the same stated rule -- so a missing key means a malformed anchors.json, not a
# case to paper over. Passing None makes load_anchors raise with
# "the contamination tripwire would silently not run" instead of scoring the whole
# field against a literal nobody chose in this file.
DEFAULT_T_IMPLAUSIBLE = None

# The agent log is scanned FIRST. A save_pretrained dump carries a multi-megabyte
# tokenizer.json, which is ~1M normalized tokens of machine-generated vocabulary;
# scanning it first spends most of the token budget before reaching the file a
# leak is actually likely to be in. Order is the whole fix -- both roots are
# still scanned.
ARTIFACT_ROOTS = ("/logs/agent", "/app/final_model")

_ARTIFACT_SHINGLES: set[str] | None = None
_MODEL = None


def render(prompt: str, response: str) -> str:
    """The one input format. Must match research/posttrain/rm_ladder.py exactly."""
    return f"{prompt}\n\nAssistant: {response}"


def load_model(sub: Path, base: Path, public: dict, base_repo: str, min_tensors: int):
    """Integrity layers first, then the model. Cached: one model, several eval sets."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.use_deterministic_algorithms(True)

    info = {"submission_bytes": scan_dir(sub)}
    info |= check_architecture(sub, base)
    info |= check_not_other_public(sub, public, base_repo)
    info |= check_lineage(sub, base, min_tensors)

    tok = AutoTokenizer.from_pretrained(str(sub))
    # Forced, not inherited: right truncation would cut the response, which is
    # the half of the input being judged.
    tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(str(sub))
    model.eval()
    if model.config.num_labels != 1:
        raise Reject(f"num_labels is {model.config.num_labels}, expected 1: the "
                     "submission must emit one scalar reward per response")
    _MODEL = (model, tok, info)
    return _MODEL


def score_texts(model, tok, prompts, responses):
    import numpy as np
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH_SIZE):
            texts = [render(p, r) for p, r in
                     zip(prompts[i:i + BATCH_SIZE], responses[i:i + BATCH_SIZE])]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LEN)
            logits = model(**enc).logits
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise Reject(f"expected logits of shape (n,1), got {tuple(logits.shape)}")
            out.append(logits.squeeze(-1).float().numpy())
    s = np.concatenate(out)
    if not np.all(np.isfinite(s)):
        raise Reject("model produced non-finite scores")
    return s


def score_eval_set(name: str, sub: Path, base: Path, test_csv: Path, public: dict,
                   base_repo: str, min_tensors: int) -> dict:
    import numpy as np
    import pandas as pd

    model, tok, info = load_model(sub, base, public, base_repo, min_tensors)
    df = pd.read_csv(test_csv)
    for col in ("prompt", "chosen", "rejected"):
        if col not in df.columns:
            raise Reject(f"internal: {test_csv} has no {col} column")
    p = df["prompt"].astype(str).tolist()
    sc = score_texts(model, tok, p, df["chosen"].astype(str).tolist())
    sr = score_texts(model, tok, p, df["rejected"].astype(str).tolist())

    wins = float((sc > sr).sum())
    ties = float((sc == sr).sum())
    acc = (wins + 0.5 * ties) / len(df)
    return dict(info) | {"acc": round(acc, 6), "n_test": len(df),
                         "ties": int(ties), "score_std": round(float(np.std(
                             np.concatenate([sc, sr]))), 6)}


# ------------------------------------------------- test-set contamination

def load_fingerprint(path: Path) -> dict:
    """Per-eval-set hashes of held-out-only word windows. Fails closed.

    instruction.md rule 1 promises the agent that the verifier checks for
    contamination. If this file is missing the check silently does not run while
    the promise still stands, which is exactly the failure the anchor loader was
    hardened against.
    """
    if not path.exists():
        raise RuntimeError(
            f"fingerprint missing at {path}; the contamination check "
            "instruction.md promises cannot run and the verifier image is broken")
    fp = json.loads(path.read_text())
    if not fp or not all(fp.values()):
        raise RuntimeError(f"fingerprint at {path} is empty for some eval set")
    return {k: set(v) for k, v in fp.items()}


def check_contamination(name: str, private: set[str]) -> dict:
    global _ARTIFACT_SHINGLES
    if _ARTIFACT_SHINGLES is None:
        _ARTIFACT_SHINGLES = artifact_shingles(ARTIFACT_ROOTS)
    overlap = private & _ARTIFACT_SHINGLES
    info = {"artifact_shingles_seen": len(_ARTIFACT_SHINGLES),
            "private_shingle_overlap": len(overlap)}
    if overlap:
        raise Reject(
            f"{len(overlap)} held-out {name} word-windows found in the submitted "
            "artifacts; these occur in the private split and nowhere in the "
            "training file shipped to the agent")
    return info


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/app/final_model")
    ap.add_argument("--base", default="/grader/base_model")
    ap.add_argument("--private", default="/grader/private")
    ap.add_argument("--anchors", default="/grader/private/anchors.json")
    ap.add_argument("--fingerprint", default="/grader/private/hh_fingerprint.json")
    ap.add_argument("--public-hashes", default="/grader/public_hashes.json")
    ap.add_argument("--base-repo", default="distilroberta-base")
    ap.add_argument("--out", default="/logs/verifier/reward.json")
    ap.add_argument("--metrics-out", default="/logs/verifier/metrics.json")
    args = ap.parse_args()

    out, metrics_out = Path(args.out), Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        anchors = load_anchors(Path(args.anchors), "acc", (), DEFAULT_T_IMPLAUSIBLE)
        fingerprint = load_fingerprint(Path(args.fingerprint))
        missing = set(anchors) - set(fingerprint)
        if missing:
            raise RuntimeError(
                f"no fingerprint for {sorted(missing)}; the contamination check "
                "would silently skip those eval sets")
        public = load_public_hashes(Path(args.public_hashes))
        min_tensors = int(MIN_TENSOR_RATIO * count_body_tensors(Path(args.base)))
        if min_tensors < 10:
            raise RuntimeError(
                f"base at {args.base} exposes too few body tensors "
                f"({min_tensors} after the ratio); the lineage floor would be "
                "trivially clearable")
    except BaseException as exc:  # noqa: BLE001
        # Config failures are not the agent's doing, so they report as a grader
        # error rather than as a rejected submission - but they still write a
        # reward, because a missing reward.json is a trial error.
        import traceback

        out.write_text(json.dumps({"reward": 0.0}))
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(
            {"eval_sets": {}, "status": "grader_error", "reward": 0.0,
             "reason": f"{type(exc).__name__}: {exc}",
             "traceback": traceback.format_exc()[-1500:]}, indent=2))
        print(f"grader_error: {type(exc).__name__}: {exc}")
        return 0

    grade_eval_sets(
        anchors,
        lambda name, anc: score_eval_set(
            name, Path(args.submission), Path(args.base),
            Path(args.private) / f"{name}_test.csv", public, args.base_repo,
            min_tensors),
        metric="acc",
        out=out,
        metrics_out=metrics_out,
        default_t_implausible=DEFAULT_T_IMPLAUSIBLE,
        postcheck=lambda name, anc, res: check_contamination(name, fingerprint[name]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
