"""End-to-end oracle run for mol-property-adapt: agent phase, then verifier phase.

Everything before this has tested pieces. modal_verify_hardening.py grades models
it constructs itself; the anchors came from standalone research scripts on GPU.
Neither answers the question that actually matters before shipping:

    does running solution/solve.sh inside the real agent image, on CPU, under the
    real budget, and grading the result with the real verifier image, reproduce
    the anchors -- and therefore score ~1.0?

reference_auc is *defined* as what solution/train_reference.py produces. If the
oracle does not land near reward 1.0 the anchor is not reproducible from its own
definition, no matter how carefully it was measured elsewhere.

Two phases, matching the task's own two containers and passing artifacts between
them exactly as Harbor does -- task.toml copies /app/final_model and
/logs/agent/train_log.txt across, and nothing else:

  agent     environment/Dockerfile, CPU-only, 8 cores, no private data present.
            Runs solution/solve.sh, which trains both eval sets.
  verifier  tests/Dockerfile, holds the private split and the anchors, runs
            tests/test.sh unmodified.

The two phases share only a Volume, so the verifier cannot see the agent's
container and the agent never sees the private split.

Expected: recovery ~1.0 on both eval sets, since the oracle is the reference
recipe. Anything well below means the anchors and the shipped solution disagree.

Run:  modal run sciml-protein-regression/scripts/modal_e2e_mol.py
"""

from __future__ import annotations

import modal

TASK = "tasks/mol-property-adapt"

agent_image = (
    modal.Image.from_dockerfile(f"{TASK}/environment/Dockerfile",
                                context_dir=f"{TASK}/environment")
    .add_local_dir(f"{TASK}/solution", "/solution")
)
verifier_image = modal.Image.from_dockerfile(f"{TASK}/tests/Dockerfile",
                                             context_dir=f"{TASK}/tests")

artifacts = modal.Volume.from_name("mol-e2e-artifacts", create_if_missing=True)
app = modal.App("mol-e2e")


@app.function(image=agent_image, cpu=8, memory=16384, timeout=14400,
              volumes={"/artifacts": artifacts})
def agent_phase() -> dict:
    """The 4-hour CPU-only budget, spent on the oracle recipe."""
    import shutil
    import subprocess
    import time
    from pathlib import Path

    # The private split must not be reachable from here. Checked rather than
    # assumed, because the whole reward rests on it.
    leaks = [str(p) for pat in ("*test*", "*inchikey*", "*anchor*")
             for p in Path("/app").rglob(pat)]
    if leaks:
        return {"status": "LEAK", "leaks": leaks}

    t0 = time.time()
    proc = subprocess.run(["bash", "/solution/solve.sh"], capture_output=True,
                          text=True, timeout=14000)
    elapsed = time.time() - t0

    out = Path("/artifacts")
    shutil.rmtree(out / "final_model", ignore_errors=True)
    shutil.rmtree(out / "logs", ignore_errors=True)
    if Path("/app/final_model").is_dir():
        shutil.copytree("/app/final_model", out / "final_model")
    if Path("/logs/agent").is_dir():
        shutil.copytree("/logs/agent", out / "logs")
    artifacts.commit()

    produced = sorted(p.name for p in (out / "final_model").iterdir()) \
        if (out / "final_model").is_dir() else []
    tail = proc.stdout[-2500:]
    print(tail, flush=True)
    return {"status": "ok" if proc.returncode == 0 else "solve_failed",
            "exit_code": proc.returncode, "elapsed_sec": round(elapsed, 1),
            "eval_sets_produced": produced, "stdout_tail": tail,
            "stderr_tail": proc.stderr[-1500:]}


@app.function(image=verifier_image, cpu=8, memory=16384, timeout=3600,
              volumes={"/artifacts": artifacts})
def verifier_phase() -> dict:
    """The real verifier image running the real entrypoint, no network."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    artifacts.reload()
    shutil.rmtree("/app/final_model", ignore_errors=True)
    shutil.rmtree("/logs/agent", ignore_errors=True)
    Path("/app").mkdir(parents=True, exist_ok=True)
    if Path("/artifacts/final_model").is_dir():
        shutil.copytree("/artifacts/final_model", "/app/final_model")
    if Path("/artifacts/logs").is_dir():
        shutil.copytree("/artifacts/logs", "/logs/agent")

    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    for stale in ("reward.json", "reward.txt", "metrics.json"):
        (logs / stale).unlink(missing_ok=True)

    proc = subprocess.run(["bash", "/tests/test.sh"], capture_output=True,
                          text=True, timeout=3000)
    out = {"exit_code": proc.returncode}
    for name in ("reward.json", "metrics.json"):
        p = logs / name
        out[name] = json.loads(p.read_text()) if p.exists() else None
    out["stderr_tail"] = proc.stderr[-1000:]
    return out


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("=== agent phase (CPU-only, real image, real solve.sh) ===")
    a = agent_phase.remote()
    print(f"status={a['status']} exit={a.get('exit_code')} "
          f"elapsed={a.get('elapsed_sec')}s")
    print(f"eval sets produced: {a.get('eval_sets_produced')}")
    if a["status"] == "LEAK":
        print(f"ABORT: private data reachable from the agent image: {a['leaks']}")
        return
    if a.get("stderr_tail", "").strip():
        print(f"agent stderr: {a['stderr_tail'][-400:]}")

    print("\n=== verifier phase (real image, no network) ===")
    v = verifier_phase.remote()
    print(f"exit={v['exit_code']}  reward.json={json.dumps(v['reward.json'])}")

    m = v["metrics.json"] or {}
    rows = []
    print(f"\n{'eval set':<10}{'status':<12}{'auc':<10}{'recovery':<11}"
          f"{'raw':<11}{'overlap':<9}")
    for name, e in (m.get("eval_sets") or {}).items():
        rows.append({"eval_set": name, **e})
        print(f"{name:<10}{e.get('status', '-'):<12}"
              f"{e.get('auc', float('nan')):<10.4f}"
              f"{e.get('recovery', float('nan')):<11.4f}"
              f"{e.get('recovery_raw', float('nan')):<11.4f}"
              f"{str(e.get('private_test_overlap', '-')):<9}")
        if e.get("reason"):
            print(f"          reason: {e['reason'][:100]}")

    dest = Path(__file__).resolve().parent / "e2e_mol.json"
    dest.write_text(json.dumps({"agent": a, "verifier": v}, indent=2) + "\n")
    print(f"\nwrote {dest}")

    reward = (v["reward.json"] or {}).get("reward")
    print("=" * 70)
    print(f"end-to-end oracle reward: {reward}")
    if reward is None:
        print("=> no reward written: the always-write guarantee failed.")
    elif reward >= 0.9:
        print("=> the shipped solution reproduces the anchors it defines.")
    elif reward >= 0.5:
        print("=> partial: one eval set reproduces, the other does not. The anchors "
              "and the shipped solution disagree on at least one set.")
    else:
        print("=> the oracle does not reproduce its own anchors. reference_auc is "
              "defined as what train_reference.py produces, so either the anchor "
              "or the script is wrong.")
