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
import re
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

# The base MLM exposes 53 encoder tensors once heads are excluded, so a genuine
# derivative compares 53. The old floor of 10 was far too loose: the comparison
# loop skips every tensor that does not match the base by name and shape, so an
# unrelated checkpoint carrying a handful of copied embedding tensors cleared it.
MIN_ENCODER_TENSORS = 50

# AUC above which a score indicates test-set contamination rather than skill.
# Anchors here are base 0.6341 / reference 0.7019 and the best fine-tune observed
# across 25 seeded runs is 0.7111, while published ChemBERTa numbers on tox21 sit
# near 0.74-0.78 on easier splits. A model trained on the recovered test rows
# scores ~0.99. 0.85 therefore sits far above anything an honest run has produced
# and far below contamination. It is a tripwire for a human to review, not proof.
DEFAULT_T_IMPLAUSIBLE = 0.85

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
        # A single non-finite element makes the norm NaN, and `NaN == 0` is False,
        # so the zero-norm guard below does not skip it. The cosine is then NaN,
        # and `NaN < worst` is also False -- so the tensor never becomes the
        # minimum and simply drops out of the floor check while still counting
        # toward `compared`. An attacker NaNs precisely the tensors that would
        # score badly and the remaining ones carry the check. Reject outright.
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise Reject(f"non-finite weights in {key}")
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        cos = float(a @ b / (na * nb))
        if not np.isfinite(cos):
            raise Reject(f"non-finite cosine on {key}")
        compared += 1
        if cos < worst:
            worst, worst_key = cos, key

    if compared < MIN_ENCODER_TENSORS:
        raise Reject(f"only {compared} encoder tensors comparable against the base "
                     f"(need {MIN_ENCODER_TENSORS}); submission does not appear to "
                     "be derived from it")
    if worst < COS_FLOOR:
        raise Reject(f"encoder lineage check failed: min per-tensor cosine "
                     f"{worst:.4f} < {COS_FLOOR} on {worst_key}")
    return {"min_tensor_cosine": round(worst, 6), "min_tensor_name": worst_key,
            "tensors_compared": compared}


# ------------------------------------------------- test-set contamination

# Artifacts the agent ships that could carry molecules. task.toml copies exactly
# these two paths across, so this is the whole reachable surface.
ARTIFACT_ROOTS = ("/app/final_model", "/logs/agent")
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
            if "".join(p.suffixes[-2:]) not in SCANNABLE_SUFFIXES and \
               p.suffix not in SCANNABLE_SUFFIXES:
                continue
            try:
                if p.stat().st_size > MAX_SCAN_BYTES:
                    continue
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

def load_anchors(path: Path) -> dict:
    """Read the anchors, failing closed.

    There is deliberately no fallback. A silent default would regrade the whole
    field against a bar nobody chose while still reporting status "ok" -- an
    unattributable scoring change with no signal. A missing or malformed anchors
    file is a broken verifier image, not a default.

    The ordering check matters as much as the presence check: reward divides by
    (reference - base), so a reversed or equal pair would send every submission
    to 0 or 1 with no error raised anywhere.
    """
    if not path.exists():
        raise RuntimeError(f"anchors.json missing at {path}; verifier image is broken")
    anchors = json.loads(path.read_text())
    if not anchors:
        raise RuntimeError(f"anchors.json at {path} defines no eval sets")
    for name, anc in anchors.items():
        for field in ("base_auc", "reference_auc", "n_tasks"):
            if field not in anc:
                raise RuntimeError(f"anchors.json[{name}] is missing {field}")
        base, ref = float(anc["base_auc"]), float(anc["reference_auc"])
        if not 0.0 < base < ref < 1.0:
            raise RuntimeError(
                f"implausible anchors for {name}: base={base} reference={ref}; "
                "expected 0 < base < reference < 1")
        t_imp = float(anc.get("t_implausible", DEFAULT_T_IMPLAUSIBLE))
        if t_imp <= ref:
            raise RuntimeError(
                f"t_implausible={t_imp} for {name} is not above reference={ref}; "
                "the tripwire would fire on legitimate work")
    return anchors


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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"eval_sets": {}, "status": "ok"}
    reward = 0.0

    try:
        anchors = load_anchors(Path(args.anchors))
        private_keys = load_private_keys(Path(args.test_keys))
        missing = set(anchors) - set(private_keys)
        if missing:
            raise RuntimeError(
                f"no held-out InChIKeys for {sorted(missing)}; the contamination "
                "check would silently skip those eval sets")
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
                entry |= check_contamination(name, private_keys[name])
                t_imp = float(anc.get("t_implausible", DEFAULT_T_IMPLAUSIBLE))
                if res["auc"] >= t_imp:
                    raise Reject(
                        f"implausible AUC {res['auc']:.4f} >= {t_imp}: no honest run "
                        "has scored here; flagged for review, not scored")
                denom = anc["reference_auc"] - anc["base_auc"]
                rec = 0.0 if denom <= 0 else (res["auc"] - anc["base_auc"]) / denom
                # `recovery_raw` is recorded uncapped on purpose. Clipping is what
                # hid a miscalibrated reference for as long as it did: every
                # fine-tune was landing at recovery 1.47 and the clip flattened
                # them all to 1.0, so the anchor being 0.04 too low was invisible
                # in the reward. The capped value is what scores; the raw value is
                # what tells you the anchors have drifted.
                entry |= {"recovery": round(min(max(rec, 0.0), 1.0), 6),
                          "recovery_raw": round(rec, 6), "status": "ok"}
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
