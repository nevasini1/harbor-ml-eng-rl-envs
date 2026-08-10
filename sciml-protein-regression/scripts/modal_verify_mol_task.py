"""Build both mol-property-adapt images on Modal and smoke-test the grader.

Until now neither image could build: tests/Dockerfile does `COPY grader/ /grader/`
and environment/Dockerfile does `COPY data/ /app/data/`, and neither directory
existed. spike/assemble_task.py generates them, but it silently skipped the
ChemBERTa base fixture (which had no builder), so even a successful assembly left
/grader/base_model absent -- read_config would raise, every eval set would floor,
and the reward would be 0.0 for every submission including the oracle.

This confirms the fix end to end, without touching local Docker:
  1. both Dockerfiles build via Image.from_dockerfile
  2. the agent image holds the training data and nothing private
  3. the verifier scores a missing submission as a clean 0.0 (always-write)
  4. the verifier scores the unmodified base model without crashing -- the path
     that was silently broken, since it exercises /grader/base_model

Run:  modal run sciml-protein-regression/scripts/modal_verify_mol_task.py
"""

from __future__ import annotations

import modal

TASK = "tasks/mol-property-adapt"

agent_image = modal.Image.from_dockerfile(
    f"{TASK}/environment/Dockerfile", context_dir=f"{TASK}/environment"
)
verifier_image = modal.Image.from_dockerfile(
    f"{TASK}/tests/Dockerfile", context_dir=f"{TASK}/tests"
)

app = modal.App("mol-task-verify")


@app.function(image=agent_image, timeout=900)
def check_agent() -> dict:
    """The agent image must have the data and must not have the answers."""
    from pathlib import Path

    out: dict = {}
    data = Path("/app/data")
    out["data_files"] = sorted(p.name for p in data.iterdir()) if data.is_dir() else []
    out["train_rows"] = (
        sum(1 for _ in open(data / "tox21_train.csv")) - 1
        if (data / "tox21_train.csv").exists() else None
    )
    # Anything answer-bearing reachable by the agent is a leak.
    leaks = []
    for pat in ("*test*", "*inchikey*", "*anchor*", "*PRIVATE*"):
        leaks += [str(p) for p in Path("/app").rglob(pat)]
        leaks += [str(p) for p in Path("/").glob(f"grader/**/{pat}")]
    out["leaks"] = leaks
    base = Path("/app/base_model")
    out["base_model_files"] = sorted(p.name for p in base.iterdir()) if base.is_dir() else []
    return out


@app.function(image=verifier_image, timeout=1800, cpu=4, memory=8192)
def check_verifier(mode: str) -> dict:
    """Run the real /tests/test.sh under a given submission condition."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    # grade.py:307 scores Path(args.submission) / <eval_set>, so the model goes in
    # a per-eval-set subdirectory -- /app/final_model/tox21, not /app/final_model.
    # instruction.md:38-39 states this contract to the agent.
    sub = Path("/app/final_model")
    shutil.rmtree(sub, ignore_errors=True)
    if mode == "base_model":
        # The unmodified base as a submission: exercises /grader/base_model, the
        # path that silently floored everything when the fixture was missing.
        for es in ("tox21", "bbbp"):
            shutil.copytree("/grader/base_model", sub / es)
    # mode == "missing" leaves /app/final_model absent on purpose.

    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    for stale in ("reward.json", "reward.txt", "metrics.json"):
        (logs / stale).unlink(missing_ok=True)

    proc = subprocess.run(
        ["bash", "/tests/test.sh"], capture_output=True, text=True, timeout=1500
    )
    out: dict = {"mode": mode, "exit_code": proc.returncode}
    for name in ("reward.json", "metrics.json"):
        p = logs / name
        out[name] = json.loads(p.read_text()) if p.exists() else None
    out["grader_present"] = sorted(p.name for p in Path("/grader").iterdir())
    out["stdout_tail"] = proc.stdout[-800:]
    out["stderr_tail"] = proc.stderr[-800:]
    return out


@app.local_entrypoint()
def main():
    import json

    print("=== agent image ===")
    a = check_agent.remote()
    print(json.dumps(a, indent=2))
    assert a["data_files"], "agent image has no /app/data"
    assert not a["leaks"], f"LEAK in agent image: {a['leaks']}"

    print("\n=== verifier image ===")
    results = list(check_verifier.map(["missing", "base_model"]))
    for r in results:
        print(f"\n[{r['mode']}] exit={r['exit_code']}")
        print(f"  /grader contents: {r['grader_present']}")
        print(f"  reward.json : {r['reward.json']}")
        print(f"  metrics.json: {json.dumps(r['metrics.json'])[:400]}")
        if r["stderr_tail"].strip():
            print(f"  stderr: {r['stderr_tail'][-300:]}")
        # The always-write guarantee, on every path.
        assert r["reward.json"] is not None, f"{r['mode']}: no reward.json written"
        assert all(
            isinstance(v, (int, float)) for v in r["reward.json"].values()
        ), f"{r['mode']}: non-numeric reward.json"
        assert "base_model" in r["grader_present"], "grader has no base_model"

    print("\nboth images build; grader writes a numeric reward on every path")
