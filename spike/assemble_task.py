"""Populate the Harbor task tree from the spike artifacts.

Enforces the one invariant that matters: nothing under split/private and no anchor
index or seed may land in the agent image. The agent build context gets training
data only; the private region, the anchors and the base-model copy for the lineage
check go exclusively into the verifier build context.
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
TASK = HERE.parent / "tasks" / "mol-property-adapt"
EVAL_SETS = ["tox21"]


def main() -> None:
    agent_data = TASK / "environment" / "data"
    grader = TASK / "tests" / "grader"
    priv = grader / "private"
    for d in (agent_data, priv):
        d.mkdir(parents=True, exist_ok=True)

    for name in EVAL_SETS:
        shutil.copy2(HERE / "split" / "agent" / f"{name}_train.csv",
                     agent_data / f"{name}_train.csv")
        shutil.copy2(HERE / "split" / "private" / f"{name}_test.csv",
                     priv / f"{name}_test.csv")

    shutil.copy2(HERE / "split" / "private" / "test_inchikeys.json",
                 priv / "test_inchikeys.json")

    # anchors.json drives the reward normalization.
    measured = json.loads((HERE / "results" / "anchors_private.json").read_text())
    anchors = {
        name: {
            "n_tasks": measured[name]["n_tasks"],
            "base_auc": measured[name]["base_auc"],
            "reference_auc": measured[name]["reference_auc"],
            "base_definition": "frozen backbone + logistic probe on mean-pooled embeddings",
            "reference_definition": "tuned fine-tune (solution/train_reference.py)",
        }
        for name in EVAL_SETS
    }
    (priv / "anchors.json").write_text(json.dumps(anchors, indent=2))

    shutil.copy2(HERE / "results" / "public_hashes.json", grader / "public_hashes.json")

    base_src = HERE / "fixtures" / "base_model_chemberta"
    if base_src.is_dir():
        dest = grader / "base_model"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(base_src, dest)

    leaks = [p for p in agent_data.rglob("*") if "test" in p.name.lower()]
    if leaks:
        raise SystemExit(f"REFUSING: private-looking files in agent context: {leaks}")

    print("agent context:")
    for p in sorted(agent_data.rglob("*")):
        print(f"  {p.relative_to(TASK)}  {p.stat().st_size:,} B")
    print("verifier context:")
    for p in sorted(grader.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(TASK)}  {p.stat().st_size:,} B")
    print("\nanchors:", json.dumps(anchors, indent=2))


if __name__ == "__main__":
    main()
