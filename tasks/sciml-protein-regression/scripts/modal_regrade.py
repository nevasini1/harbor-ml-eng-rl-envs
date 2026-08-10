"""Re-score finished trials on Modal, using the real verifier image. No local Docker.

Why not just reuse scripts/regrade.sh
-------------------------------------
That script does `docker build tests/ && docker run`, which needs a local Docker
daemon. This does the same thing on Modal instead: `modal.Image.from_dockerfile`
builds `tests/Dockerfile` with `tests/` as the build context, so the graded
artifact is the genuine verifier image -- pinned torch 2.6.0, the revision-pinned
base model baked at /grader/base_model, the private split at /tests/private_test,
and the real /tests/test.sh entrypoint. This is not a Modal-shaped reimplementation
of the grader; it is the grader.

It also runs on Linux, which matters for one specific reason: Harbor gates egress
control on `sys.platform == "linux" or _egress_control_kernel_support()`
(harbor/environments/docker/docker.py:194). Docker Desktop on macOS fails that
probe, so `network_mode = "no-network"` is silently ignored there -- verified by
reaching zenodo.org from inside a local agent container. Anything that needs the
network policy to actually hold has to run on Linux.

Usage:
    modal run tasks/sciml-protein-regression/scripts/modal_regrade.py
    modal run tasks/sciml-protein-regression/scripts/modal_regrade.py --trials "a,b"

Each trial is a path (relative to repo root) containing artifacts/app/final_model.
Results are written back to <trial>/regrade_modal/.
"""

from __future__ import annotations

import modal

TASK = "tasks/sciml-protein-regression"

# Trials to re-score. Each must have artifacts/app/final_model on disk.
DEFAULT_TRIALS = [
    # known: reward 0.5, spearman 0.4312160593525205 -- the tier-0.5 regression lock
    "jobs/2026-08-09__14-08-50/sciml-protein-regression__ZnpRu34",
    # known: reward 1.0, spearman 0.535779224790035 -- frozen probe, the tier-1.0 lock
    "jobs/frozen-probe-check/frozen_head_only",
]

# The genuine verifier image, built from the task's own Dockerfile.
verifier_image = modal.Image.from_dockerfile(
    f"{TASK}/tests/Dockerfile",
    context_dir=f"{TASK}/tests",
)

app = modal.App("harbor-protein-regrade")


@app.function(image=verifier_image, timeout=3600, cpu=4, memory=8192)
def regrade(submission: dict[str, bytes], trial: str) -> dict:
    """Run the real /tests/test.sh against one submission and return its outputs."""
    import json
    import subprocess
    from pathlib import Path

    app_dir = Path("/app/final_model")
    app_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in submission.items():
        (app_dir / name).write_bytes(blob)

    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    for stale in ("reward.json", "reward.txt", "metrics.json", "reward_meta.json"):
        (logs / stale).unlink(missing_ok=True)

    proc = subprocess.run(
        ["bash", "/tests/test.sh"], capture_output=True, text=True, timeout=3300
    )

    out: dict = {"trial": trial, "exit_code": proc.returncode}
    for name in ("reward.json", "metrics.json", "reward_meta.json"):
        p = logs / name
        out[name] = json.loads(p.read_text()) if p.exists() else None
    out["reward_txt"] = (
        (logs / "reward.txt").read_text().strip()
        if (logs / "reward.txt").exists()
        else None
    )
    out["stdout_tail"] = proc.stdout[-1500:]
    out["stderr_tail"] = proc.stderr[-1500:]
    return out


@app.local_entrypoint()
def main(trials: str = ""):
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    names = [t.strip() for t in trials.split(",") if t.strip()] or DEFAULT_TRIALS

    payloads, kept = [], []
    for t in names:
        sub = root / t / "artifacts" / "app" / "final_model"
        if not sub.is_dir():
            print(f"SKIP {t} -- no artifacts/app/final_model")
            continue
        payloads.append({p.name: p.read_bytes() for p in sub.iterdir() if p.is_file()})
        kept.append(t)
        mb = sum(p.stat().st_size for p in sub.iterdir() if p.is_file()) / 1e6
        print(f"staged {t} ({mb:.1f} MB)")

    if not kept:
        print("nothing to regrade")
        return

    results = list(regrade.starmap(zip(payloads, kept)))

    print("\n" + "=" * 74)
    for r in results:
        rj = r["reward.json"] or {}
        mj = r["metrics.json"] or {}
        meta = r["reward_meta.json"] or {}
        print(f"\n{r['trial']}")
        print(f"  exit={r['exit_code']}  reward.json={rj}")
        print(f"  reason={meta.get('reason')}  reward.txt={r['reward_txt']}")
        if mj:
            print(f"  spearman={mj.get('spearman')}  t_weak={mj.get('t_weak')} "
                  f"t_strong={mj.get('t_strong')}")
            print(f"  cosine_min={mj.get('cosine_min')} "
                  f"n_tensors_compared={mj.get('n_tensors_compared')}")
        # The contract the changes must not break.
        assert rj and list(rj) == ["reward"], f"reward.json must be single-key, got {rj}"
        assert all(isinstance(v, (int, float)) for v in rj.values()), "non-numeric reward"
        if not r["stdout_tail"].strip().startswith("{"):
            print(f"  stdout: {r['stdout_tail'][-300:]}")

    dest = Path(__file__).resolve().parent / "regrade_modal.json"
    dest.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwrote {dest}")
