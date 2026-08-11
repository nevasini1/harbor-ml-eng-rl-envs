"""Crash-safe verifier for mleval/sciml-protein-regression.

Always writes reward.json. Never raises out of main().
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from verifier_core import Reject, check_lineage

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ARCH_KEYS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "tie_word_embeddings",
)

# Per-tensor cosine floor for tensors shared with the base backbone.
# Legitimate fine-tunes stay far above this; unrelated same-shape inits do not.
COS_FLOOR = 0.90

# Minimum backbone tensors that must match the base by name and shape. Every real
# submission graded so far compared 108; anything far below means the checkpoint
# does not share the base encoder and the cosine floor is being applied to a
# handful of coincidentally-matching tensors.
MIN_BACKBONE_TENSORS = 100

# Known public weight sha256s that are NOT the provided base (swap attempts).
# Base itself is allowed (frozen-backbone + head is legitimate).
FORBIDDEN_WEIGHT_SHA256 = {
    # facebook/esm2_t12_35M_UR50D model.safetensors
    "e35647818e0e064351d4531ed480d225a002567b4b2b93ad3a9246d753150fc0",
    # facebook/esm2_t30_150M_UR50D model.safetensors
    "c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566",
}


def write_reward(path: Path, payload: dict) -> None:
    """Split a payload across three channels; reward.json carries `reward` alone.

    Harbor's VerifierResult.rewards is dict[str, float|int], so a string anywhere
    in reward.json fails validation and loses the whole score.

    `reward` is also the only key that belongs there. Harbor's default dataset
    metric (harbor.metrics.base.aggregate_reward_dicts) keys off the union of
    reward names across trials: one extra key anywhere pushes the whole dataset
    into the per-key branch, which drops the single `mean` headline and averages
    every diagnostic against a fabricated 0 for tasks that never reported it.
    Numeric detail goes to metrics.json, non-numeric to reward_meta.json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric = {}
    meta = {}
    for k, v in payload.items():
        if isinstance(v, bool):
            numeric[k] = int(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                numeric[k] = 0.0
            else:
                numeric[k] = float(v) if not isinstance(v, int) else v
        else:
            meta[k] = v

    reward = numeric.pop("reward", 0.0)
    path.write_text(json.dumps({"reward": reward}, indent=2) + "\n")
    if numeric:
        (path.parent / "metrics.json").write_text(json.dumps(numeric, indent=2) + "\n")
    if meta:
        (path.parent / "reward_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (path.parent / "reward.txt").write_text(str(float(reward)) + "\n")


def fail(out: Path, reason: str, **extra) -> int:
    payload = {"reward": 0.0, "reason": reason, **extra}
    write_reward(out, payload)
    print(json.dumps(payload))
    return 0


def load_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text())


def arch_dict(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in ARCH_KEYS}


def check_architecture(sub_cfg: dict, base_cfg: dict) -> str | None:
    a, b = arch_dict(sub_cfg), arch_dict(base_cfg)
    if a != b:
        return f"architecture_mismatch: {a} != {b}"
    # HF often omits num_labels and encodes it via id2label instead.
    n_labels = sub_cfg.get("num_labels")
    if n_labels is None and isinstance(sub_cfg.get("id2label"), dict):
        n_labels = len(sub_cfg["id2label"])
    try:
        n_labels = int(n_labels)
    except (TypeError, ValueError):
        return f"num_labels_unreadable:got={sub_cfg.get('num_labels')}"
    if n_labels != 1:
        return f"num_labels_must_be_1:got={n_labels}"
    return None


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_forbidden_hashes(model_dir: Path) -> str | None:
    for p in model_dir.rglob("*.safetensors"):
        digest = file_sha256(p)
        if digest in FORBIDDEN_WEIGHT_SHA256:
            return f"forbidden_public_checkpoint:{p.name}:{digest[:12]}"
    return None


def overlapping_backbone_cosine(sub_dir: Path, base_dir: Path) -> tuple[float, int]:
    """Min per-tensor cosine over the shared backbone, via common/verifier_core.py.

    This used to be ~90 lines of its own implementation, and it carried a bug the
    shared version has since fixed: the floor was applied to the minimum over ALL
    shared tensors, including 1-D biases. A bias whose entries are near zero
    rotates a long way under a functionally irrelevant update, so an honest
    fine-tune can be rejected on a vector that does not affect the model's output.
    That cost a real agent trial on pref-reward-model a reward of 0.0 where 0.86
    was earned (commit 8d42c88). The runs recorded for this task pass at
    cosine_min 0.992-1.000, so here it was latent rather than active -- which is
    luck, not safety.

    `check_lineage` raises on failure and returns a dict; this file reports reasons
    as strings and writes numeric-only rewards, so the contract is adapted here
    rather than propagated. Returns (min weight-matrix cosine, tensors compared).

    Not migrated: check_architecture and check_forbidden_hashes. The former returns
    a reason string rather than raising, and the latter compares against a
    hardcoded set of sibling sha256s rather than the public_hashes.json index the
    shared layer expects -- so it never had the repo-prefix bug fixed in 8d42c88.
    Migrating those means changing this grader's error contract, which its 16
    hardening assertions are written against.
    """
    # Both gates are disabled here and left to the caller. check_lineage RAISES on
    # a low tensor count and on a low cosine; this grader reports those as its own
    # named reasons (`insufficient_backbone_overlap`, `low_backbone_cosine`) and
    # its 16 hardening assertions are written against those strings. Letting the
    # shared function raise first would silently rewrite them to
    # `provenance_failed:Reject:...`.
    info = check_lineage(sub_dir, base_dir, min_tensors=0, cos_floor=0.0)
    return float(info["min_tensor_cosine"]), int(info["tensors_compared"])


def predict(model_dir: Path, seqs: list[str], batch_size: int = 16) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    preds = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        logits = model(**enc).logits.squeeze(-1)
        preds.append(logits.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds).astype(np.float64)


def spearman_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_pred) < 1e-12:
        return 0.0
    rho = spearmanr(y_true, y_pred).statistic
    if rho is None or (isinstance(rho, float) and (math.isnan(rho) or math.isinf(rho))):
        return 0.0
    return float(rho)


def load_tiers() -> dict:
    """Fixed Spearman thresholds baked beside the grader (calibrated offline).

    Fails closed. The previous silent fallback to {"t_weak": 0.20} meant a build
    that dropped tiers.json regraded the whole field against a bar 0.19 lower
    while still reporting reason "ok" -- an unattributable scoring change with no
    signal. A missing thresholds file is a broken image, not a default.
    """
    path = Path(__file__).resolve().parent / "tiers.json"
    if not path.exists():
        raise RuntimeError(f"tiers.json missing at {path}; verifier image is broken")
    tiers = json.loads(path.read_text())
    t_weak, t_strong = float(tiers["t_weak"]), float(tiers["t_strong"])
    if not 0.0 < t_weak < t_strong < 1.0:
        raise RuntimeError(f"implausible tiers: t_weak={t_weak} t_strong={t_strong}")
    return tiers


def implausibility_ceiling(tiers: dict) -> float:
    """Spearman above which a score indicates test-set contamination, not skill.

    The private test set is the public FLIP2 meltome-mixed file minus the rows
    shipped to the agent, so an agent with egress can reconstruct it exactly and
    train on the answers. `network_mode = "no-network"` closes that, but Harbor
    silently ignores it on non-Linux hosts (docker.py:194), so this is the only
    host-independent check.

    The separation is wide enough to exploit. Measured legitimate results: a
    frozen probe reaches 0.546 +/- 0.005 (n=8), the best fine-tune ever observed
    on this task is 0.5709, and published ESM2-8M ceilings on meltome-mixed are
    ~0.6-0.7. A model trained on the recovered test rows scores near 0.99,
    because the public labels rank-correlate 0.9998 with the private ones.

    This is a tripwire, not proof: it flags a score no honest run has produced,
    for a human to review. Keep the margin large so it never fires on skill.
    """
    return float(tiers.get("t_implausible", 0.75))


def reward_from_spearman(rho: float, t_weak: float, t_strong: float) -> float:
    """Tiered reward: 0 / 0.5 / 1.0 vs frozen-probe and strong-oracle bars."""
    if rho < t_weak:
        return 0.0
    if rho < t_strong:
        return 0.5
    return 1.0


def maybe_overlap_check(submission: Path, private_seqs: set[str]) -> dict:
    """Optional: if agent left a train file, reject exact private overlaps."""
    candidates = [
        submission.parent / "train.csv.gz",
        submission.parent / "train.csv",
        Path("/app/train.csv.gz"),
        Path("/data/train.csv.gz"),
    ]
    for c in candidates:
        if not c.exists():
            continue
        try:
            df = pd.read_csv(c, compression="gzip" if c.suffix == ".gz" else None)
            if "sequence" not in df.columns:
                continue
            overlap = len(private_seqs & set(df["sequence"].astype(str)))
            return {"train_artifact": str(c), "overlap_with_private": overlap}
        except Exception:
            continue
    return {"overlap_with_private": None}


def grade(submission: Path, base: Path, test_csv: Path, out: Path) -> int:
    if not submission.exists():
        return fail(out, "missing_submission")

    try:
        sub_cfg = load_config(submission)
        base_cfg = load_config(base)
    except Exception as e:
        return fail(out, f"config_unreadable:{type(e).__name__}")

    err = check_architecture(sub_cfg, base_cfg)
    if err:
        return fail(out, err)

    err = check_forbidden_hashes(submission)
    if err:
        return fail(out, err)

    try:
        cos_min, n_tensors = overlapping_backbone_cosine(submission, base)
    except Exception as e:
        return fail(out, f"provenance_failed:{type(e).__name__}:{e}")

    # A count floor, not just non-zero. The loop skips any tensor whose name or
    # shape does not match the base, so a model with no lineage passes with
    # n_tensors == 1 by copying a single base tensor into an otherwise unrelated
    # checkpoint. Real fine-tunes of this architecture compare 108.
    if n_tensors < MIN_BACKBONE_TENSORS:
        return fail(
            out,
            "insufficient_backbone_overlap",
            n_tensors_compared=n_tensors,
            min_required=MIN_BACKBONE_TENSORS,
        )
    if cos_min < COS_FLOOR:
        return fail(
            out,
            "weight_provenance_failed",
            cosine_min=cos_min,
            n_tensors=n_tensors,
            cos_floor=COS_FLOOR,
        )

    try:
        df = pd.read_csv(test_csv, compression="gzip")
        seqs = df["sequence"].astype(str).tolist()
        y = df["target"].to_numpy(dtype=np.float64)
        overlap_info = maybe_overlap_check(submission, set(seqs))
        # /data/train.csv.gz is expected agent data and was built disjoint; only
        # fail if an unexpected train artifact overlaps. Skip the provided /data.
        if (
            overlap_info.get("overlap_with_private")
            and overlap_info.get("train_artifact") not in ("/data/train.csv.gz",)
            and overlap_info["overlap_with_private"] > 0
        ):
            return fail(out, "private_test_overlap_in_train_artifact", **overlap_info)

        preds = predict(submission, seqs)
        if not np.all(np.isfinite(preds)):
            return fail(
                out,
                "non_finite_predictions",
                n_non_finite=int((~np.isfinite(preds)).sum()),
                cosine_min=cos_min,
                n_tensors_compared=n_tensors,
            )
        rho = spearman_safe(y, preds)
        tiers = load_tiers()
        t_weak = float(tiers["t_weak"])
        t_strong = float(tiers["t_strong"])
        t_implausible = implausibility_ceiling(tiers)
        if rho >= t_implausible:
            return fail(
                out,
                "implausible_private_fit",
                spearman=rho,
                t_implausible=t_implausible,
                cosine_min=cos_min,
                n_tensors_compared=n_tensors,
            )
        reward = reward_from_spearman(rho, t_weak, t_strong)
    except Exception as e:
        return fail(
            out,
            f"eval_failed:{type(e).__name__}",
            detail=str(e)[:300],
            cosine_min=cos_min,
            n_tensors=n_tensors,
        )

    payload = {
        "reward": reward,
        "spearman": rho,
        "t_weak": t_weak,
        "t_strong": t_strong,
        "n_test": int(len(y)),
        "cosine_min": cos_min,
        "n_tensors_compared": n_tensors,
        "reason": "ok",
    }
    write_reward(out, payload)
    print(json.dumps(payload))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, default=Path("/app/final_model"))
    ap.add_argument("--base", type=Path, default=Path("/grader/base_model"))
    ap.add_argument("--test", type=Path, default=Path("/tests/private_test/test.csv.gz"))
    ap.add_argument("--out", type=Path, default=Path("/logs/verifier/reward.json"))
    args = ap.parse_args()
    try:
        return grade(args.submission, args.base, args.test, args.out)
    except Exception:
        traceback.print_exc()
        return fail(args.out, "grader_unhandled_exception")


if __name__ == "__main__":
    raise SystemExit(main())
