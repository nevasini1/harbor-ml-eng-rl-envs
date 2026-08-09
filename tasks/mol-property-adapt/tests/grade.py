"""Verifier for mleval/mol-property-adapt.

Contract: the agent submits /app/final_model/<eval_set>/ as a `save_pretrained`
directory loadable by AutoModelForSequenceClassification. No agent-authored code
is imported or executed here.

reward = mean over eval sets of clip((auc - base) / (reference - base), 0, 1)

Design constraints this file is written against:

  * Never crash into ambiguity. Any agent output - missing, malformed, hostile,
    wrong shape, non-finite, or a model that raises at inference - must still
    produce a reward. Every failure path floors that eval set to 0.0 and the run
    continues; the process always writes reward.json.

  * Three anti-substitution layers, because they fail on disjoint inputs:
      1. architecture-config hash vs the provided base
      2. sha256 vs known public checkpoints (rejects any other public model)
      3. per-tensor float64 cosine vs the base encoder
    Layer 3 deliberately ALLOWS an unmodified encoder: freezing the backbone and
    training only a head is a legitimate strategy, so "weights must have moved" is
    not a valid requirement.

  * float64 for cosine. float32 accumulation over a large embedding matrix can
    return values above 1. Per-tensor minimum, not a global cosine, because a
    single global figure stays high even for shuffled weights.

  * reward.json carries exactly one key. Harbor's default dataset metric raises on
    a multi-key reward dict, so per-eval-set detail goes to metrics.json instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MAX_CONFIG_BYTES = 1 << 20
MAX_SUBMISSION_BYTES = 1 << 30  # base encoder is ~14 MB; 1 GB is already absurd
MAX_SMILES_CHARS = 1024
COS_FLOOR = 0.90
BATCH_SIZE = 64

ARCH_FIELDS = [
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "type_vocab_size",
]


class Reject(Exception):
    """Submission is unusable or fails integrity; this eval set floors to 0."""


# --------------------------------------------------------------- safe reading

def scan_dir(path: Path) -> int:
    if not path.is_dir():
        raise Reject("submission path is not a directory")
    total = 0
    for p in path.rglob("*"):
        if p.is_symlink():
            raise Reject(f"symlink in submission: {p.name}")
        if p.is_file():
            total += p.stat().st_size
            if total > MAX_SUBMISSION_BYTES:
                raise Reject("submission exceeds size cap")
    return total


def read_config(path: Path) -> dict:
    cfg = path / "config.json"
    if not cfg.is_file():
        raise Reject("missing config.json")
    if cfg.stat().st_size > MAX_CONFIG_BYTES:
        raise Reject("config.json implausibly large")
    try:
        return json.loads(cfg.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError) as exc:
        raise Reject(f"unparseable config.json: {type(exc).__name__}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------- layers

def check_architecture(sub: Path, base: Path) -> dict:
    s, b = read_config(sub), read_config(base)
    sig = lambda c: hashlib.sha256(  # noqa: E731
        json.dumps({k: c.get(k) for k in ARCH_FIELDS}, sort_keys=True).encode()
    ).hexdigest()
    if sig(s) != sig(b):
        diff = {k: (b.get(k), s.get(k)) for k in ARCH_FIELDS if b.get(k) != s.get(k)}
        raise Reject(f"architecture differs from base: {diff}")
    return {"arch_sha256": sig(s)}


def check_not_other_public(sub: Path, public: dict, base_repo: str) -> dict:
    hits = []
    for f in sorted(sub.glob("*.safetensors")) + sorted(sub.glob("*.bin")):
        d = sha256_file(f)
        for ref, ref_digest in public.items():
            if d == ref_digest and not ref.startswith(base_repo):
                hits.append({"file": f.name, "matches": ref})
    if hits:
        raise Reject(f"weights are bit-identical to another public checkpoint: {hits}")
    return {"public_checkpoint_matches": 0}


def load_tensors(path: Path) -> dict:
    """Weights as numpy arrays, from safetensors or a .bin.

    The provided base ships pytorch_model.bin while `save_pretrained` emits
    safetensors, so both sides must handle either format. `.bin` is read with
    weights_only=True: it is a pickle, and the submission side is untrusted.
    """
    import numpy as np

    st = path / "model.safetensors"
    if st.is_file():
        from safetensors import safe_open

        out = {}
        with safe_open(str(st), framework="np") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out

    bin_path = path / "pytorch_model.bin"
    if bin_path.is_file():
        import torch

        try:
            raw = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise Reject(f"unreadable pytorch_model.bin: {type(exc).__name__}")
        return {k: v.detach().numpy() for k, v in raw.items()
                if hasattr(v, "detach") and v.dtype.is_floating_point}

    raise Reject("no model.safetensors or pytorch_model.bin in submission")


def normalize_key(k: str) -> str:
    """Strip the task-model prefix so encoder tensors line up across checkpoints.

    The base is an MLM (`roberta.*`) and the submission a sequence classifier
    (`roberta.*` too, but `lm_head.*` vs `classifier.*` differ). Comparing on the
    shared encoder body is the point; heads legitimately have no counterpart.
    """
    for prefix in ("roberta.", "bert.", "esm.", "model."):
        if k.startswith(prefix):
            return k[len(prefix):]
    return k


def check_lineage(sub: Path, base: Path) -> dict:
    import numpy as np

    sub_t = {normalize_key(k): v for k, v in load_tensors(sub).items()}
    base_t = {normalize_key(k): v for k, v in load_tensors(base).items()}

    worst, worst_key, compared = 1.0, None, 0
    for key, b in base_t.items():
        # Encoder tensors only; a fresh task head has nothing to compare against.
        if key not in sub_t or key.startswith(("lm_head", "classifier")):
            continue
        a = sub_t[key]
        if a.shape != b.shape:
            raise Reject(f"tensor shape mismatch on {key}: {a.shape} vs {b.shape}")
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        cos = float(a @ b / (na * nb))
        compared += 1
        if cos < worst:
            worst, worst_key = cos, key

    if compared < 10:
        raise Reject(f"only {compared} encoder tensors comparable against the base; "
                     "submission does not appear to be derived from it")
    if worst < COS_FLOOR:
        raise Reject(f"encoder lineage check failed: min per-tensor cosine "
                     f"{worst:.4f} < {COS_FLOOR} on {worst_key}")
    return {"min_tensor_cosine": round(worst, 6), "min_tensor_name": worst_key,
            "tensors_compared": compared}


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


def score_eval_set(sub: Path, base: Path, test_csv: Path, public: dict,
                   base_repo: str, n_tasks: int) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.use_deterministic_algorithms(True)

    info = {"submission_bytes": scan_dir(sub)}
    info |= check_architecture(sub, base)
    info |= check_not_other_public(sub, public, base_repo)
    info |= check_lineage(sub, base)

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
    ap.add_argument("--public-hashes", default="/grader/public_hashes.json")
    ap.add_argument("--base-repo", default="DeepChem/ChemBERTa-77M-MLM")
    ap.add_argument("--out", default="/logs/verifier/reward.json")
    ap.add_argument("--metrics-out", default="/logs/verifier/metrics.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"eval_sets": {}, "status": "ok"}
    reward = 0.0

    try:
        anchors = json.loads(Path(args.anchors).read_text())
        public_raw = json.loads(Path(args.public_hashes).read_text())
        public = {
            f"{repo}/{fname}": meta["sha256"]
            for repo, entry in public_raw.items()
            for fname, meta in entry.get("files", {}).items()
            if meta.get("sha256")
        }

        recoveries = []
        for name, anc in anchors.items():
            entry: dict = {"base_auc": anc["base_auc"], "reference_auc": anc["reference_auc"]}
            try:
                res = score_eval_set(
                    Path(args.submission) / name,
                    Path(args.base),
                    Path(args.private) / f"{name}_test.csv",
                    public,
                    args.base_repo,
                    anc["n_tasks"],
                )
                entry |= res
                denom = anc["reference_auc"] - anc["base_auc"]
                rec = 0.0 if denom <= 0 else (res["auc"] - anc["base_auc"]) / denom
                entry |= {"recovery": round(min(max(rec, 0.0), 1.0), 6), "status": "ok"}
            except Reject as exc:
                entry |= {"recovery": 0.0, "status": "rejected", "reason": str(exc)}
            except BaseException as exc:  # noqa: BLE001
                entry |= {"recovery": 0.0, "status": "error",
                          "reason": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-1500:]}
            metrics["eval_sets"][name] = entry
            recoveries.append(entry["recovery"])

        reward = sum(recoveries) / len(recoveries) if recoveries else 0.0
    except BaseException as exc:  # noqa: BLE001
        metrics |= {"status": "grader_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:]}
        reward = 0.0

    metrics["reward"] = round(reward, 6)
    # Exactly one key: Harbor's default dataset metric rejects multi-key rewards.
    out.write_text(json.dumps({"reward": round(reward, 6)}))
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
