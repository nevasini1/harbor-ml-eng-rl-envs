"""Regression-test the four hardening changes made to the mol grader.

Why
---
modal_verify_mol_task.py smoke-tests that both images build and that the grader
always writes a numeric reward. It does not exercise the checks added in this
session, and two of those changed control flow on the *accept* path:

    MIN_ENCODER_TENSORS  10 -> 50
    non-finite weights   now rejected outright

If the base exposes fewer comparable tensors than I measured, or if a legitimate
save_pretrained round-trip renames anything, that floor rejects honest work. A
verifier that fails closed on real submissions is worse than the loophole it
replaced, so the accept path is tested first and treated as the primary result.

Four cases, one per change, each asserting a specific outcome rather than "no
crash":

  legit        AutoModelForSequenceClassification derived from the base, head
               untrained. Encoder identical, so cosine is 1.0 and every encoder
               tensor is comparable. MUST be accepted, and MUST compare 53.
  nan_weight   legit, with one element of one encoder tensor set to NaN. The old
               code let this through: the cosine is NaN, `NaN < worst` is False,
               so the tensor silently dropped out of the floor while still
               counting toward the total. MUST now be rejected.
  one_tensor   a checkpoint carrying a single base encoder tensor. Passed the old
               `compared < 10` floor at compared == 1. MUST now be rejected.
  no_anchors   anchors.json deleted from the verifier image. The old protein-side
               equivalent fell back to an uncalibrated threshold and still
               reported ok. MUST now produce grader_error, name the missing file,
               and still write a numeric reward.json.
  contaminated legit, with the held-out tox21 CSV planted inside the submission.
               instruction.md rule 1 promises the verifier checks for this; until
               now it did not. MUST be rejected on InChIKey overlap.
  no_test_keys test_inchikeys.json deleted. The promised check cannot run, so it
               MUST fail closed rather than silently skip.

Lineage runs before the model is loaded for inference (score_eval_set checks
architecture, public hashes, then lineage, and only then calls from_pretrained),
so the crafted cases do not need to be loadable.

Run:  modal run tasks/sciml-protein-regression/scripts/modal_verify_hardening.py
"""

from __future__ import annotations

import modal

TASK = "tasks/mol-property-adapt"
EXPECTED_ENCODER_TENSORS = 53

verifier_image = modal.Image.from_dockerfile(
    f"{TASK}/tests/Dockerfile", context_dir=f"{TASK}/tests"
)

app = modal.App("mol-grader-hardening")


@app.function(image=verifier_image, timeout=1800, cpu=4, memory=8192)
def run_case(mode: str) -> dict:
    import json
    import shutil
    import subprocess
    from pathlib import Path

    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base = Path("/grader/base_model")
    sub = Path("/app/final_model")
    shutil.rmtree(sub, ignore_errors=True)
    dest = sub / "tox21"

    anchors = Path("/grader/private/anchors.json")
    anchors_backup = None

    if mode == "no_anchors":
        anchors_backup = anchors.read_text()
        anchors.unlink()

    keys_path = Path("/grader/private/test_inchikeys.json")
    keys_backup = None
    if mode == "no_test_keys":
        keys_backup = keys_path.read_text()
        keys_path.unlink()

    # A legitimate submission for each eval set: same encoder, fresh head.
    tok = AutoTokenizer.from_pretrained(str(base))
    for eval_set, n_lab in (("tox21", 12), ("bbbp", 1)):
        d = sub / eval_set
        AutoModelForSequenceClassification.from_pretrained(
            str(base), num_labels=n_lab,
            problem_type="multi_label_classification").save_pretrained(str(d))
        tok.save_pretrained(str(d))

    if mode == "contaminated":
        # The answers, left where the agent would have left them.
        shutil.copy("/grader/private/tox21_test.csv", dest / "train_data.csv")

    if mode == "nan_weight":
        wf = dest / "model.safetensors"
        t = load_file(str(wf))
        key = "roberta.encoder.layer.0.attention.self.query.weight"
        assert key in t, f"{key} not in submission; keys start {sorted(t)[:3]}"
        t[key] = t[key].clone()
        t[key][0, 0] = float("nan")
        save_file(t, str(wf))

    if mode == "one_tensor":
        wf = dest / "model.safetensors"
        t = load_file(str(wf))
        keep = "roberta.embeddings.word_embeddings.weight"
        crafted = {keep: t[keep].clone(),
                   "classifier.dense.weight": t["classifier.dense.weight"].clone(),
                   "classifier.out_proj.weight": t["classifier.out_proj.weight"].clone()}
        save_file(crafted, str(wf))

    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    for stale in ("reward.json", "reward.txt", "metrics.json"):
        (logs / stale).unlink(missing_ok=True)

    proc = subprocess.run(["bash", "/tests/test.sh"], capture_output=True,
                          text=True, timeout=1500)

    if anchors_backup is not None:
        anchors.write_text(anchors_backup)
    if keys_backup is not None:
        keys_path.write_text(keys_backup)

    out = {"mode": mode, "exit_code": proc.returncode}
    for name in ("reward.json", "metrics.json"):
        p = logs / name
        out[name] = json.loads(p.read_text()) if p.exists() else None
    out["stderr_tail"] = proc.stderr[-500:]
    return out


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    results = {r["mode"]: r for r in run_case.map(
        ["legit", "nan_weight", "one_tensor", "no_anchors",
         "contaminated", "no_test_keys"])}

    checks, failures = [], []

    def check(label, ok, detail):
        checks.append({"check": label, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            failures.append(label)

    print("\n=== every path still writes a numeric reward ===")
    for mode, r in results.items():
        rj = r["reward.json"]
        check(f"{mode}: reward.json numeric",
              rj is not None and all(isinstance(v, (int, float)) for v in rj.values()),
              json.dumps(rj))

    print("\n=== accept path (the risk introduced by MIN_ENCODER_TENSORS) ===")
    legit = results["legit"]["metrics.json"]["eval_sets"]["tox21"]
    check("legit submission is not rejected",
          legit.get("status") == "ok",
          f"status={legit.get('status')} reason={legit.get('reason', '-')}")
    check(f"legit compares {EXPECTED_ENCODER_TENSORS} encoder tensors",
          legit.get("tensors_compared") == EXPECTED_ENCODER_TENSORS,
          f"tensors_compared={legit.get('tensors_compared')}")
    check("legit cosine is 1.0 (encoder untouched)",
          legit.get("min_tensor_cosine", 0) > 0.999,
          f"min_tensor_cosine={legit.get('min_tensor_cosine')}")
    check("legit records uncapped recovery_raw",
          "recovery_raw" in legit,
          f"recovery_raw={legit.get('recovery_raw')} recovery={legit.get('recovery')}")

    print("\n=== reject paths (the loopholes that were open) ===")
    nan = results["nan_weight"]["metrics.json"]["eval_sets"]["tox21"]
    check("NaN weight is rejected",
          nan.get("status") == "rejected" and "non-finite" in nan.get("reason", ""),
          f"status={nan.get('status')} reason={nan.get('reason', '-')[:90]}")

    one = results["one_tensor"]["metrics.json"]["eval_sets"]["tox21"]
    check("single-tensor lineage is rejected",
          one.get("status") == "rejected" and "encoder tensors" in one.get("reason", ""),
          f"status={one.get('status')} reason={one.get('reason', '-')[:90]}")

    print("\n=== contamination (instruction.md rule 1) ===")
    check("legit submission passes the contamination check",
          "private_test_overlap" in legit and legit["private_test_overlap"] == 0,
          f"overlap={legit.get('private_test_overlap')} "
          f"molecules_seen={legit.get('artifact_molecules_seen')}")
    con = results["contaminated"]["metrics.json"]["eval_sets"]["tox21"]
    check("planted held-out molecules are rejected",
          con.get("status") == "rejected" and "held-out" in con.get("reason", ""),
          f"status={con.get('status')} reason={con.get('reason', '-')[:90]}")

    print("\n=== fail-closed anchors ===")
    na = results["no_anchors"]["metrics.json"]
    check("missing anchors.json is a grader_error, not a default",
          na.get("status") == "grader_error" and "anchors.json missing" in na.get("reason", ""),
          f"status={na.get('status')} reason={na.get('reason', '-')[:90]}")

    nk = results["no_test_keys"]["metrics.json"]
    check("missing test_inchikeys.json fails closed",
          nk.get("status") == "grader_error" and "test_inchikeys" in nk.get("reason", ""),
          f"status={nk.get('status')} reason={nk.get('reason', '-')[:90]}")

    dest = Path(__file__).resolve().parent / "grader_hardening_check.json"
    dest.write_text(json.dumps(
        {"checks": checks, "failures": failures, "raw": results}, indent=2) + "\n")

    print("\n" + "=" * 68)
    print(f"{len(checks) - len(failures)}/{len(checks)} checks passed")
    print(f"wrote {dest}")
    if failures:
        print("\nFAILED: " + ", ".join(failures))
        print("The accept-path failures matter most: a grader that rejects honest "
              "submissions is worse than the loophole it replaced.")
    else:
        print("\n=> hardening verified: loopholes closed, honest submissions unaffected.")
