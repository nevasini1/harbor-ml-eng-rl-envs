"""Verifier for mleval/qa-sft-adapt.

Contract: the agent submits `/app/final_model` -- a `save_pretrained` directory
loadable by AutoModelForCausalLM, plus its tokenizer. No agent-authored code is
imported or executed here.

reward = mean over eval sets of clip((acc - base) / (reference - base), 0, 1)

`acc` is multiple-choice accuracy measured the way an LM eval harness measures
it: each candidate answer is appended to a fixed prompt, the model's mean
log-probability over the answer tokens is read off, and the highest-scoring
candidate is the prediction. Nothing is generated and nothing is sampled, so the
metric is a deterministic function of the weights -- which is what makes it
usable as a reward at all. A generation-plus-judge metric would put a second
model's noise inside the reward signal.

Two things here that the mol grader does not have:

  * a **fourth integrity layer** (`check_closer_to_base`). This base has a
    publicly released instruction-tuned sibling with an identical architecture.
    Starting from that sibling instead of the base passes the config-hash check
    (same config), passes the sha256 check (the weights moved during the agent's
    own training) and passes the cosine floor (the sibling is itself derived from
    the base). Only a comparison *between* candidate ancestors separates them.

  * length-normalized scoring. Without it the metric is dominated by answer
    length: a one-word distractor outranks a correct four-word answer on most
    items, and the task measures brevity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from textmatch import artifact_shingles
from verifier_core import (Reject, check_architecture, check_closer_to_base,
                           check_lineage, check_not_other_public,
                           count_body_tensors, grade_eval_sets, load_anchors,
                           load_public_hashes, scan_dir)

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MAX_LEN = 160
BATCH_SIZE = 32
MIN_TENSOR_RATIO = 0.9

# Accuracy above which a score indicates the held-out items were seen rather than
# knowledge. The measured reference is 0.55-0.60 depending on the eval set and a
# model trained on the recovered test rows scores >0.95; models 50x this size
# report 0.75-0.85 on these corpora. A tripwire for a human to review, not proof.
DEFAULT_T_IMPLAUSIBLE = 0.90

# The agent log is scanned FIRST. A save_pretrained dump carries a multi-megabyte
# tokenizer.json, which is ~1M normalized tokens of machine-generated vocabulary;
# scanning it first spends most of the token budget before reaching the file a
# leak is actually likely to be in. Order is the whole fix -- both roots are
# still scanned.
ARTIFACT_ROOTS = ("/logs/agent", "/app/final_model")

_ARTIFACT_SHINGLES: set[str] | None = None
_MODEL = None


# ---------------------------------------------------------------- rendering

def render_prompt(question: str) -> str:
    """The one prompt format. Must match spike/posttrain/sft_ladder.py exactly."""
    return f"Question: {question.strip()}\nAnswer:"


def render_choice(choice: str) -> str:
    return f" {choice.strip()}"


# ------------------------------------------------------------------ scoring

def load_model(sub: Path, base: Path, public: dict, base_repo: str,
               min_tensors: int, siblings: dict):
    """Integrity layers first, then the model. Cached across eval sets."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Thread count is pinned by the image, the model is in eval mode with no
    # dropout, and nothing is sampled -- so scoring is already reproducible.
    # `torch.use_deterministic_algorithms(True)` is deliberately NOT set here:
    # it raises on any op without a deterministic kernel, and a raise inside the
    # grader floors an honest submission to 0. The mol grader can afford the flag
    # because it runs a 3-layer RoBERTa; this one runs a 30-layer Llama.
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    info = {"submission_bytes": scan_dir(sub)}
    info |= check_architecture(sub, base)
    info |= check_not_other_public(sub, public, base_repo)
    info |= check_lineage(sub, base, min_tensors)
    if siblings:
        info |= check_closer_to_base(sub, base, siblings)

    tok = AutoTokenizer.from_pretrained(str(sub))
    model = AutoModelForCausalLM.from_pretrained(str(sub))
    model.eval()
    _MODEL = (model, tok, info)
    return _MODEL


def choice_logprobs(model, tok, questions, choices_per_q):
    """Mean log-probability per answer token, for every candidate answer."""
    import numpy as np
    import torch

    flat, owner = [], []
    for qi, (q, chs) in enumerate(zip(questions, choices_per_q)):
        for ch in chs:
            flat.append((render_prompt(q), render_choice(ch)))
            owner.append(qi)

    out = np.full(len(flat), -np.inf)
    with torch.no_grad():
        for i in range(0, len(flat), BATCH_SIZE):
            batch = flat[i:i + BATCH_SIZE]
            ids, starts = [], []
            for prompt, cont in batch:
                p = tok(prompt, add_special_tokens=False)["input_ids"]
                c = tok(cont, add_special_tokens=False)["input_ids"]
                if not c:
                    c = [tok.eos_token_id]
                seq = (p + c)[-MAX_LEN:]
                starts.append(max(len(seq) - len(c), 1))
                ids.append(seq)
            width = max(len(s) for s in ids)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            inp = torch.full((len(ids), width), pad, dtype=torch.long)
            att = torch.zeros((len(ids), width), dtype=torch.long)
            for j, s in enumerate(ids):
                inp[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            logits = model(input_ids=inp, attention_mask=att).logits.float()
            if logits.ndim != 3 or logits.shape[1] != width:
                raise Reject(f"expected logits of shape (n,{width},vocab), "
                             f"got {tuple(logits.shape)}")
            if not torch.all(torch.isfinite(logits)):
                raise Reject("model produced non-finite logits")
            logprobs = torch.log_softmax(logits, dim=-1)
            for j, s in enumerate(ids):
                st = starts[j]
                tgt = torch.tensor(s[st:])
                lp = logprobs[j, st - 1:len(s) - 1, :].gather(
                    1, tgt.unsqueeze(1)).squeeze(1)
                out[i + j] = float(lp.mean())

    per_q: list[list[float]] = [[] for _ in questions]
    for k, qi in enumerate(owner):
        per_q[qi].append(out[k])
    return per_q


def score_eval_set(name: str, sub: Path, base: Path, test_csv: Path, public: dict,
                   base_repo: str, min_tensors: int, siblings: dict) -> dict:
    import numpy as np
    import pandas as pd

    model, tok, info = load_model(sub, base, public, base_repo, min_tensors, siblings)
    df = pd.read_csv(test_csv)
    for col in ("question", "choices", "answer_idx"):
        if col not in df.columns:
            raise Reject(f"internal: {test_csv} has no {col} column")
    qs = df["question"].astype(str).tolist()
    chs = [json.loads(c) for c in df["choices"]]
    gold = df["answer_idx"].to_numpy()

    scores = choice_logprobs(model, tok, qs, chs)
    # argmax breaks ties toward the lowest index, which is a fixed position in a
    # shuffled choice list, so a degenerate model lands at chance rather than at
    # an artificially high or low number.
    pred = np.array([int(np.argmax(s)) for s in scores])
    acc = float((pred == gold).mean())
    return dict(info) | {"acc": round(acc, 6), "n_test": len(df),
                         "n_choices_mean": round(float(np.mean(
                             [len(c) for c in chs])), 3)}


# ------------------------------------------------- test-set contamination

def load_fingerprint(path: Path) -> dict:
    """Per-eval-set hashes of held-out-only word windows. Fails closed."""
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
    ap.add_argument("--siblings", default="/grader/siblings")
    ap.add_argument("--private", default="/grader/private")
    ap.add_argument("--anchors", default="/grader/private/anchors.json")
    ap.add_argument("--fingerprint", default="/grader/private/qa_fingerprint.json")
    ap.add_argument("--public-hashes", default="/grader/public_hashes.json")
    ap.add_argument("--base-repo", default="HuggingFaceTB/SmolLM2-135M")
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
        # Fails closed as well: this base has a same-architecture public sibling,
        # so a verifier image built without it would silently drop the only layer
        # that catches a laundered instruct checkpoint.
        sib_root = Path(args.siblings)
        siblings = {p.name: p for p in sorted(sib_root.iterdir())
                    if p.is_dir()} if sib_root.is_dir() else {}
        if not siblings:
            raise RuntimeError(
                f"no sibling checkpoints at {sib_root}; check_closer_to_base "
                "cannot run and the verifier image is broken")
    except BaseException as exc:  # noqa: BLE001
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
            min_tensors, siblings),
        metric="acc",
        out=out,
        metrics_out=metrics_out,
        default_t_implausible=DEFAULT_T_IMPLAUSIBLE,
        postcheck=lambda name, anc, res: check_contamination(name, fingerprint[name]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
