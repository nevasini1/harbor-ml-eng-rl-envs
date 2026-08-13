"""Verifier machinery shared by every task in this repo.

Extracted verbatim from `tasks/mol-property-adapt/tests/grade.py`, which is the
only grader here whose behaviour has been measured end to end through Harbor.
Nothing was redesigned on the way out: the mol grader now imports these functions
and reproduces its recorded reward (0.909661 on `jobs/mol-oracle-modal`) to the
sixth decimal. That check is the reason the extraction is trustworthy, and it is
the check to re-run after touching this file.

What is here is everything that is *not* about a particular dataset:

  * three anti-substitution layers, which fail on disjoint inputs --
    architecture-config hash, sha256 against known public checkpoints, and a
    per-tensor float64 cosine against the base encoder;
  * `load_anchors`, which fails closed rather than defaulting;
  * `recovery`, the normalization between two measured anchors;
  * `grade_eval_sets`, the driver that guarantees a reward is always written and
    that no single eval set can take the process down with it.

What is deliberately *not* here: how a metric is computed, and what counts as
contamination. Both are dataset-specific -- InChIKey overlap for molecules,
normalized-text overlap for prompts -- and pretending otherwise would produce an
abstraction that each task has to fight.

Every task's `tests/verifier_core.py` is a byte-identical copy of this file,
placed there because a Docker build context cannot reach outside itself.
`common/sync.py --check` fails if any copy has drifted.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MAX_CONFIG_BYTES = 1 << 20
MAX_SUBMISSION_BYTES = 1 << 30
COS_FLOOR = 0.90

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

# Task heads are legitimately new weights with no counterpart in the base, so
# they are excluded from the lineage comparison rather than failing it.
HEAD_PREFIXES = ("lm_head", "classifier", "score", "cls.")


class Reject(Exception):
    """Submission is unusable or fails integrity; this eval set floors to 0."""


# --------------------------------------------------------------- safe reading

def scan_dir(path: Path, max_bytes: int = MAX_SUBMISSION_BYTES) -> int:
    if not path.is_dir():
        raise Reject("submission path is not a directory")
    total = 0
    for p in path.rglob("*"):
        if p.is_symlink():
            raise Reject(f"symlink in submission: {p.name}")
        if p.is_file():
            total += p.stat().st_size
            if total > max_bytes:
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


# ----------------------------------------------------------------- layer 1/3

def check_architecture(sub: Path, base: Path, fields: list[str] | None = None) -> dict:
    fields = fields or ARCH_FIELDS
    s, b = read_config(sub), read_config(base)
    sig = lambda c: hashlib.sha256(  # noqa: E731
        json.dumps({k: c.get(k) for k in fields}, sort_keys=True).encode()
    ).hexdigest()
    if sig(s) != sig(b):
        diff = {k: (b.get(k), s.get(k)) for k in fields if b.get(k) != s.get(k)}
        raise Reject(f"architecture differs from base: {diff}")
    return {"arch_sha256": sig(s)}


# ----------------------------------------------------------------- layer 2/3

def check_not_other_public(sub: Path, public: dict, base_repo: str) -> dict:
    """Reject weights bit-identical to a public checkpoint that is not the base.

    The repo is compared for equality, not as a string prefix. A prefix test
    looks equivalent and is not: `HuggingFaceTB/SmolLM2-135M-Instruct` starts
    with `HuggingFaceTB/SmolLM2-135M`, so an unmodified instruct checkpoint --
    the single most attractive substitution for the SFT task, since it shares the
    base's architecture exactly -- would have been treated as the base itself and
    waved through this layer.
    """
    hits = []
    for f in sorted(sub.glob("*.safetensors")) + sorted(sub.glob("*.bin")):
        d = sha256_file(f)
        for ref, meta in public.items():
            if d == meta["sha256"] and meta["repo"] != base_repo:
                hits.append({"file": f.name, "matches": ref})
    if hits:
        raise Reject(f"weights are bit-identical to another public checkpoint: {hits}")
    return {"public_checkpoint_matches": 0}


def load_public_hashes(path: Path) -> dict:
    """Pinned-checkpoint index as `repo/file -> {repo, sha256}`.

    The repo is carried as its own field rather than being recovered by splitting
    the key: a Hub repo can serve files from subdirectories, so the last path
    component is not reliably the filename.
    """
    raw = json.loads(Path(path).read_text())
    return {
        f"{repo}/{fname}": {"repo": repo, "sha256": meta["sha256"]}
        for repo, entry in raw.items()
        for fname, meta in entry.get("files", {}).items()
        if meta.get("sha256")
    }


# ----------------------------------------------------------------- layer 3/3

def load_tensors(path: Path) -> dict:
    """Weights as float32 numpy arrays, from safetensors or a .bin.

    Some pinned bases ship `pytorch_model.bin` while `save_pretrained` emits
    safetensors, so both sides must handle either format. `.bin` is read with
    weights_only=True: it is a pickle, and the submission side is untrusted.

    safetensors is read through torch, not through `framework="np"`. numpy has no
    bfloat16, and `framework="np"` raises `TypeError: data type 'bfloat16' not
    understood` on any checkpoint stored in it -- which SmolLM2-135M is, and
    which a `torch_dtype=torch.bfloat16` fine-tune of any base would be. Reading
    via torch and upcasting to float32 keeps every comparison exact for float32
    checkpoints while making bf16 ones work at all.
    """
    import torch

    st = path / "model.safetensors"
    if st.is_file():
        from safetensors import safe_open

        out = {}
        with safe_open(str(st), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if t.is_floating_point():
                    out[k] = t.to(torch.float32).numpy()
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

    A base MLM stores `roberta.*` / `model.*` and a submitted task model stores
    the same body under the same prefix but a different head. Comparing on the
    shared body is the point; heads legitimately have no counterpart.
    """
    for prefix in ("roberta.", "bert.", "esm.", "model.", "transformer."):
        if k.startswith(prefix):
            return k[len(prefix):]
    return k


def check_lineage(sub: Path, base: Path, min_tensors: int,
                  cos_floor: float = COS_FLOOR,
                  head_prefixes: tuple[str, ...] = HEAD_PREFIXES) -> dict:
    """Per-tensor float64 cosine between the submission's body and the base's.

    float64 because float32 accumulation over a large embedding matrix can return
    values above 1. Per-tensor minimum, not a global cosine, because a single
    global figure stays high even for shuffled weights.

    This layer deliberately ALLOWS an unmodified body: freezing the backbone and
    training only a head is a legitimate strategy, so "weights must have moved"
    is not a valid requirement. It rejects a *different* model, not a lazy one --
    the anchors are what make laziness score zero.
    """
    import numpy as np

    sub_t = {normalize_key(k): v for k, v in load_tensors(sub).items()}
    base_t = {normalize_key(k): v for k, v in load_tensors(base).items()}

    # The floor is applied to weight MATRICES only (ndim >= 2), not to 1-D bias
    # and LayerNorm vectors. This is not a loosening; it is a correction, and a
    # real agent run is what exposed it.
    #
    # `codex` fine-tuned the provided base honestly on pref-reward-model and was
    # rejected at reward 0.0: min per-tensor cosine 0.8541 on
    # `encoder.layer.0.attention.self.key.bias`. Measured over that submission,
    # all 51 weight matrices sat at cosine >= 0.9999 (median 1.0000) and exactly
    # 1 of 100 tensors was under the floor -- a 768-element bias whose entries are
    # near zero, so a functionally irrelevant update rotates the vector a long way.
    # Attention key biases are the extreme case: they barely affect the output at
    # all, which is why several implementations omit them.
    #
    # Cosine is a similarity measure for direction, and direction is only
    # informative when the vector has substance. Every adversarial case this layer
    # exists for moves weight matrices -- a shuffled embedding, a substituted
    # encoder -- so restricting the floor costs nothing: the `shuffled` fixture is
    # still caught at cosine 0.007 on a 2-D embedding. 1-D cosines are still
    # computed and reported, just not used to reject.
    worst, worst_key, compared = 1.0, None, 0
    worst_1d, worst_1d_key = 1.0, None
    for key, b in base_t.items():
        if key not in sub_t or key.startswith(head_prefixes):
            continue
        a = sub_t[key]
        if a.shape != b.shape:
            raise Reject(f"tensor shape mismatch on {key}: {a.shape} vs {b.shape}")
        is_matrix = b.ndim >= 2
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
        if is_matrix:
            if cos < worst:
                worst, worst_key = cos, key
        elif cos < worst_1d:
            worst_1d, worst_1d_key = cos, key

    if compared < min_tensors:
        raise Reject(f"only {compared} encoder tensors comparable against the base "
                     f"(need {min_tensors}); submission does not appear to "
                     "be derived from it")
    if worst < cos_floor:
        raise Reject(f"encoder lineage check failed: min per-tensor cosine "
                     f"{worst:.4f} < {cos_floor} on {worst_key} (weight matrices "
                     "only; 1-D vectors are reported, not gated)")
    return {"min_tensor_cosine": round(worst, 6), "min_tensor_name": worst_key,
            "min_vector_cosine": round(worst_1d, 6), "min_vector_name": worst_1d_key,
            "tensors_compared": compared}


def check_closer_to_base(sub: Path, base: Path, siblings: dict[str, Path],
                         head_prefixes: tuple[str, ...] = HEAD_PREFIXES) -> dict:
    """Reject a submission that resembles a *sibling* checkpoint more than the base.

    The fourth layer, and the only one that catches the substitution the other
    three cannot: a publicly released model with the **same architecture** as the
    base -- an instruction-tuned variant of it, typically -- lightly fine-tuned
    before submission. The architecture hash matches (same config), the sha256
    check passes (the weights were moved), and the cosine floor passes (the
    sibling is itself derived from the base, so it stays correlated with it).

    What separates the two cases is direction, not distance: an honest run starts
    at the base and moves a little, so it stays closer to the base than to any
    sibling; a laundered instruct checkpoint is closer to the sibling it started
    from. Mean per-tensor cosine, not the minimum, because the minimum is set by
    whichever single tensor moved most and is far noisier.

    Costs one copy of each sibling in the verifier image. Only worth it where a
    same-architecture public sibling actually exists.
    """
    import numpy as np

    def mean_cos(a_t: dict, b_t: dict) -> tuple[float, int]:
        vals = []
        for key, b in b_t.items():
            if key not in a_t or key.startswith(head_prefixes):
                continue
            a = a_t[key]
            if a.shape != b.shape:
                continue
            a = a.astype(np.float64).ravel()
            b = b.astype(np.float64).ravel()
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                raise Reject(f"non-finite weights in {key}")
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                continue
            vals.append(float(a @ b / (na * nb)))
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    sub_t = {normalize_key(k): v for k, v in load_tensors(sub).items()}
    base_cos, n_base = mean_cos(sub_t, {normalize_key(k): v
                                        for k, v in load_tensors(base).items()})
    info = {"mean_cosine_to_base": round(base_cos, 6), "sibling_cosines": {}}
    for name, path in siblings.items():
        sib_t = {normalize_key(k): v for k, v in load_tensors(Path(path)).items()}
        sib_cos, n_sib = mean_cos(sub_t, sib_t)
        info["sibling_cosines"][name] = round(sib_cos, 6)
        if n_sib and sib_cos > base_cos:
            raise Reject(
                f"submission is closer to the public checkpoint {name} "
                f"(mean cosine {sib_cos:.6f}) than to the provided base "
                f"({base_cos:.6f}); it does not appear to start from the base")
    if not n_base:
        raise Reject("no tensors comparable against the base")
    return info


def count_body_tensors(base: Path,
                       head_prefixes: tuple[str, ...] = HEAD_PREFIXES) -> int:
    """How many tensors `check_lineage` can compare for a perfect derivative.

    A task sets `min_tensors` as a fraction of this rather than as a constant, so
    the floor survives a change of base checkpoint instead of silently becoming
    either unreachable or trivial.
    """
    return sum(1 for k in load_tensors(base)
               if not normalize_key(k).startswith(head_prefixes))


# ------------------------------------------------------------------- anchors

def eval_set_items(doc: dict) -> dict:
    """The eval sets in an anchors document, with sidecar metadata removed.

    `anchors.json` carries `_criterion` and `_provenance` alongside its eval sets.
    Anything reading the file directly -- rather than through `load_anchors`, which
    filters them itself -- must go through this, or it treats `_criterion` as an eval
    set. That is not hypothetical: adding those two keys broke
    `research/posttrain/verify_graders.py`, which did
    `sorted(json.loads(...))` and then looked for `_criterion_test.csv`, and
    `research/plot_criterion.py`, which indexed `a["reference_auc"]` on the
    criterion block. Both failures were mine, both shipped, and neither was caught
    by a checker that only reads these files without running their consumers.
    """
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def load_anchors(path: Path, metric: str, extra_required: tuple[str, ...] = (),
                 default_t_implausible: float | None = None) -> dict:
    """Read the anchors, failing closed.

    There is deliberately no fallback. A silent default would regrade the whole
    field against a bar nobody chose while still reporting status "ok" -- an
    unattributable scoring change with no signal. A missing or malformed anchors
    file is a broken verifier image, not a default.

    The ordering check matters as much as the presence check: reward divides by
    (reference - base), so a reversed or equal pair would send every submission
    to 0 or 1 with no error raised anywhere.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"anchors.json missing at {path}; verifier image is broken")
    anchors = json.loads(path.read_text())
    # Keys beginning with "_" are sidecar metadata, not eval sets: `_criterion`
    # records the rule these anchors were screened by and `_provenance` records the
    # commit and script that wrote them. They are separated here, at the one place
    # that reads this file, so that provenance can live beside the numbers it
    # describes instead of in a second file that can drift from them. Every
    # remaining key must be a real eval set and is validated as one.
    sidecar = {k: v for k, v in anchors.items() if k.startswith("_")}
    anchors = {k: v for k, v in anchors.items() if not k.startswith("_")}
    if not anchors:
        raise RuntimeError(
            f"anchors.json at {path} defines no eval sets"
            + (f" (only sidecar keys {sorted(sidecar)})" if sidecar else ""))
    bk, rk = f"base_{metric}", f"reference_{metric}"
    for name, anc in anchors.items():
        for field in (bk, rk, *extra_required):
            if field not in anc:
                raise RuntimeError(f"anchors.json[{name}] is missing {field}")
        base, ref = float(anc[bk]), float(anc[rk])
        if not 0.0 < base < ref < 1.0:
            raise RuntimeError(
                f"implausible anchors for {name}: base={base} reference={ref}; "
                "expected 0 < base < reference < 1")
        t_imp = anc.get("t_implausible", default_t_implausible)
        if t_imp is None:
            raise RuntimeError(
                f"anchors.json[{name}] has no t_implausible and the task set no "
                "default; the contamination tripwire would silently not run")
        if float(t_imp) <= ref:
            raise RuntimeError(
                f"t_implausible={t_imp} for {name} is not above reference={ref}; "
                "the tripwire would fire on legitimate work")
    return anchors


def recovery(value: float, base: float, reference: float) -> tuple[float, float]:
    """(capped, raw) normalized recovery between two measured anchors.

    The raw value is returned so the caller can record it. Clipping is what hid a
    miscalibrated reference on the mol task for an entire working session: every
    fine-tune landed at recovery 1.47 and the clip flattened them all to 1.0, so
    an anchor 0.04 too low was invisible in the reward.
    """
    denom = reference - base
    raw = 0.0 if denom <= 0 else (value - base) / denom
    return min(max(raw, 0.0), 1.0), raw


# -------------------------------------------------------------------- driver

def grade_eval_sets(
    anchors: dict,
    score_one: Callable[[str, dict], dict],
    *,
    metric: str,
    out: Path,
    metrics_out: Path,
    default_t_implausible: float | None = None,
    postcheck: Callable[[str, dict, dict], dict] | None = None,
) -> dict:
    """Score every eval set, aggregate, and always write a reward.

    `score_one(name, anchor)` returns a dict containing `metric`; `postcheck`
    (contamination, typically) runs after it and may raise `Reject`. Any failure
    -- missing, malformed, hostile, non-finite, or a model that raises at
    inference -- floors that eval set to 0.0 and the run continues.

    reward.json carries the aggregate plus one score per eval set. It used to carry
    exactly one key, on the belief that "Harbor's default dataset metric raises on a
    multi-key reward dict" -- which the vendored Harbor contradicts: its
    `_parse_reward_json` is a bare `json.loads` and `aggregate_reward_dicts` has an
    explicit multi-key branch that preserves every key. Per-eval-set DIAGNOSTICS
    still go to metrics.json; only scores belong in the reward channel.
    """
    out, metrics_out = Path(out), Path(metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"eval_sets": {}, "status": "ok"}
    reward = 0.0
    bk, rk = f"base_{metric}", f"reference_{metric}"

    try:
        scores = []
        for name, anc in anchors.items():
            entry: dict = {bk: anc[bk], rk: anc[rk]}
            try:
                res = score_one(name, anc)
                entry |= res
                if postcheck is not None:
                    entry |= postcheck(name, anc, res)
                t_imp = float(anc.get("t_implausible", default_t_implausible))
                if res[metric] >= t_imp:
                    raise Reject(
                        f"implausible {metric} {res[metric]:.4f} >= {t_imp}: no "
                        "honest run has scored here; flagged for review, not scored")
                cap, raw = recovery(res[metric], float(anc[bk]), float(anc[rk]))
                entry |= {"recovery": round(cap, 6), "recovery_raw": round(raw, 6),
                          "status": "ok"}
            except Reject as exc:
                entry |= {"recovery": 0.0, "status": "rejected", "reason": str(exc)}
            except BaseException as exc:  # noqa: BLE001
                entry |= {"recovery": 0.0, "status": "error",
                          "reason": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-1500:]}
            metrics["eval_sets"][name] = entry
            scores.append(entry["recovery"])
        reward = sum(scores) / len(scores) if scores else 0.0
    except BaseException as exc:  # noqa: BLE001
        metrics |= {"status": "grader_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:]}
        reward = 0.0

    metrics["reward"] = round(reward, 6)

    # reward.json is the aggregate PLUS one score per eval set, which is the shape
    # Harbor aggregates: `aggregate_reward_dicts` keeps every key and means each
    # across trials, so `reward` stays primary and the per-eval-set scores become
    # additional metrics. Single-key output collapsed to `{"mean": 0.734}` in the
    # job record, which threw away the most interesting thing a multi-eval-set task
    # produces -- qa-sft-adapt's 0.867 / 0.908 / 0.427, where the shortfall is
    # concentrated on the hardest set, was invisible to anything reading Harbor.
    #
    # Scores only, all on the reward's own [0,1] scale. Diagnostics -- raw metrics,
    # cosines, tensor counts, shingle overlap, status and reason -- stay in
    # metrics.json. This is the distinction the earlier single-key rule missed: what
    # broke was putting COUNTS here (an `n_test` of 3427 aggregated as a metric),
    # not carrying more than one key.
    payload = {"reward": round(reward, 6)}
    for name, entry in metrics["eval_sets"].items():
        if name == "reward":
            raise RuntimeError(
                "an eval set is named 'reward', which would overwrite the aggregate "
                "in reward.json; rename it")
        payload[name] = round(float(entry.get("recovery", 0.0)), 6)
    out.write_text(json.dumps(payload))
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics
