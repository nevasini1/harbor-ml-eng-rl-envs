"""Crash-safe verifier prototype.

Contract: the agent submits a `save_pretrained` directory loadable by
AutoModelForSequenceClassification with num_labels=1. No agent-authored code is
executed here.

Reward is normalized recovery between two anchors:

    reward = clip((rho_sub - rho_base) / (rho_ref - rho_base), 0, 1)

Every failure path still emits a reward. The verifier assumes the submission is
arbitrary and potentially hostile: it never imports from the submission
directory, bounds what it reads before reading it, and treats any exception as
a zero rather than letting it escape.

Anti-substitution runs three independent layers, because they fail on disjoint
inputs:
  1. architecture-config hash   - catches a larger ESM (35M/150M/650M) swapped in
  2. sha256 against public HF   - catches any unmodified public checkpoint
  3. per-tensor cosine vs base  - catches a same-shape model of different lineage

Layer 3 deliberately ALLOWS an unmodified backbone: freezing the encoder and
training only a head is a legitimate strategy here, so "weights must have moved"
is not a valid requirement.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import traceback
from pathlib import Path

# Keep the grader deterministic before torch/numpy import threading decisions.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

MAX_CONFIG_BYTES = 1 << 20      # a real HF config is ~800 B
MAX_SUBMISSION_BYTES = 2 << 30  # base is 31 MB; anything near 2 GB is not a fine-tune
COS_FLOOR = 0.90

ARCH_FIELDS = [
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "position_embedding_type",
]


class RejectionError(Exception):
    """Submission is invalid or fails an integrity check; reward floors to 0."""


# ------------------------------------------------------------ safe file access

def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_symlink():
            raise RejectionError(f"symlink in submission: {p.name}")
        if p.is_file():
            total += p.stat().st_size
            if total > MAX_SUBMISSION_BYTES:
                raise RejectionError("submission exceeds size cap")
    return total


def load_config(path: Path) -> dict:
    cfg = path / "config.json"
    if not cfg.is_file():
        raise RejectionError("missing config.json")
    if cfg.stat().st_size > MAX_CONFIG_BYTES:
        raise RejectionError("config.json implausibly large")
    try:
        return json.loads(cfg.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise RejectionError(f"unparseable config.json: {type(exc).__name__}")


def arch_signature(cfg: dict) -> str:
    import hashlib

    canon = json.dumps({k: cfg.get(k) for k in ARCH_FIELDS}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------- integrity layers

def check_architecture(sub_dir: Path, base_dir: Path) -> dict:
    sub, base = load_config(sub_dir), load_config(base_dir)
    sub_sig, base_sig = arch_signature(sub), arch_signature(base)
    if sub_sig != base_sig:
        diff = {k: (base.get(k), sub.get(k)) for k in ARCH_FIELDS if base.get(k) != sub.get(k)}
        raise RejectionError(f"architecture mismatch vs base: {diff}")
    return {"arch_sha256": sub_sig}


def check_not_public(sub_dir: Path, public_hashes: dict[str, str], base_repo: str) -> dict:
    """Reject if any submitted weight file is bit-identical to a public checkpoint
    other than the provided base."""
    matches = []
    for weight in sorted(sub_dir.glob("*.safetensors")) + sorted(sub_dir.glob("*.bin")):
        digest = file_sha256(weight)
        for repo_file, repo_digest in public_hashes.items():
            if digest == repo_digest and not repo_file.startswith(base_repo):
                matches.append({"file": weight.name, "matched": repo_file})
    if matches:
        raise RejectionError(f"submitted weights match a public checkpoint: {matches}")
    return {"public_hash_matches": 0}


def check_lineage(sub_dir: Path, base_dir: Path) -> dict:
    """Per-tensor cosine against the base, in float64.

    float32 accumulation is not safe here: on a 155M-element embedding matrix it
    can return cosine > 1. Comparison is per-tensor because a single global
    cosine is inflated by 1-D norm vectors even for unrelated weights.
    """
    import numpy as np
    from safetensors import safe_open

    sub_file = sub_dir / "model.safetensors"
    base_file = base_dir / "model.safetensors"
    if not sub_file.is_file():
        raise RejectionError("missing model.safetensors")

    worst, checked, skipped = 1.0, 0, 0
    worst_name = None
    with safe_open(str(sub_file), framework="np") as sf, \
         safe_open(str(base_file), framework="np") as bf:
        base_keys = set(bf.keys())
        for key in bf.keys():
            # Only compare encoder tensors; a fine-tune legitimately adds a head.
            if key not in sf.keys():
                skipped += 1
                continue
            a = sf.get_tensor(key).astype(np.float64).ravel()
            b = bf.get_tensor(key).astype(np.float64).ravel()
            if a.shape != b.shape:
                raise RejectionError(f"tensor shape mismatch on {key}")
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                continue
            cos = float(np.dot(a, b) / (na * nb))
            checked += 1
            if cos < worst:
                worst, worst_name = cos, key

    if checked == 0:
        raise RejectionError("no comparable tensors against base")
    if worst < COS_FLOOR:
        raise RejectionError(
            f"weight lineage check failed: min per-tensor cosine {worst:.4f} "
            f"< {COS_FLOOR} on {worst_name}"
        )
    return {
        "min_tensor_cosine": round(worst, 6),
        "min_tensor_name": worst_name,
        "tensors_compared": checked,
        "tensors_missing_from_submission": skipped,
        "base_tensor_count": len(base_keys),
    }


# ------------------------------------------------------------------- scoring

def score_submission(sub_dir: Path, test_csv: Path, max_len: int, batch_size: int) -> float:
    import numpy as np
    import pandas as pd
    import torch
    from scipy.stats import spearmanr
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    opener = gzip.open if test_csv.suffix == ".gz" else open
    with opener(test_csv, "rt") as fh:
        df = pd.read_csv(fh)
    seqs = df["sequence"].str.slice(0, max_len).tolist()
    y = df["target"].to_numpy()

    tok = AutoTokenizer.from_pretrained(str(sub_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(sub_dir), num_labels=1)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            enc = tok(seqs[i : i + batch_size], return_tensors="pt",
                      padding=True, truncation=True, max_length=max_len)
            out = model(**enc).logits
            if out.ndim != 2 or out.shape[1] != 1:
                raise RejectionError(f"expected logits of shape (n,1), got {tuple(out.shape)}")
            preds.append(out.squeeze(-1).float().numpy())

    pred = np.concatenate(preds)
    if not np.all(np.isfinite(pred)):
        raise RejectionError("submission produced non-finite predictions")
    if np.allclose(pred, pred[0]):
        return 0.0  # constant output has undefined rank correlation; treat as no signal
    rho = spearmanr(pred, y).statistic
    return 0.0 if (rho is None or math.isnan(rho)) else float(rho)


# ---------------------------------------------------------------------- main

def grade(args) -> dict:
    sub_dir = Path(args.submission)
    base_dir = Path(args.base)

    if not sub_dir.is_dir():
        raise RejectionError("submission path is not a directory")
    size = dir_size(sub_dir)

    report: dict = {"submission_bytes": size}
    report |= check_architecture(sub_dir, base_dir)

    public = json.loads(Path(args.public_hashes).read_text()) if args.public_hashes else {}
    flat = {
        f"{repo}/{name}": meta["sha256"]
        for repo, entry in public.items()
        for name, meta in entry.get("files", {}).items()
        if meta.get("sha256")
    }
    report |= check_not_public(sub_dir, flat, args.base_repo)
    report |= check_lineage(sub_dir, base_dir)

    rho = score_submission(sub_dir, Path(args.test), args.max_len, args.batch_size)
    denom = args.ref_spearman - args.base_spearman
    recovery = 0.0 if denom <= 0 else (rho - args.base_spearman) / denom
    report |= {
        "spearman": round(rho, 6),
        "base_spearman": args.base_spearman,
        "ref_spearman": args.ref_spearman,
        "reward": round(min(max(recovery, 0.0), 1.0), 6),
        "status": "ok",
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--public-hashes", default=None)
    ap.add_argument("--base-repo", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--base-spearman", type=float, required=True)
    ap.add_argument("--ref-spearman", type=float, required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        report = grade(args)
    except RejectionError as exc:
        report = {"reward": 0.0, "status": "rejected", "reason": str(exc)}
    except BaseException as exc:  # noqa: BLE001 - a verifier must never propagate
        report = {
            "reward": 0.0,
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }

    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
