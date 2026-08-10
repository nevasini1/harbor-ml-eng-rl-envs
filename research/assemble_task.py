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
EVAL_SETS = ["tox21", "bbbp"]


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
    # Carried through, not restated. The definitions were previously hardcoded
    # here and drifted: they still said "logistic probe on mean-pooled
    # embeddings" long after both anchors had been re-measured on the CLS head
    # the verifier actually accepts.
    carry = ("n_tasks", "base_auc", "reference_auc", "t_implausible",
             "base_definition", "reference_definition", "band", "band_sigma")
    anchors = {}
    for name in EVAL_SETS:
        m = measured[name]
        missing = [k for k in ("n_tasks", "base_auc", "reference_auc",
                               "t_implausible") if k not in m]
        if missing:
            raise SystemExit(
                f"REFUSING: anchors_private.json[{name}] is missing {missing}. "
                "grade.py fails closed on these, so a partial anchor file would "
                "produce a verifier that cannot run.")
        anchors[name] = {k: m[k] for k in carry if k in m}
    (priv / "anchors.json").write_text(json.dumps(anchors, indent=2))

    shutil.copy2(HERE / "results" / "public_hashes.json", grader / "public_hashes.json")

    # Fail loudly. This used to be `if base_src.is_dir(): copy`, and the fixture
    # has never existed, so every assembly silently produced a grader with no
    # /grader/base_model. tests/Dockerfile's `COPY grader/ /grader/` then failed
    # to build; had it built, read_config would raise on the missing base, every
    # eval set would floor, and the reward would be 0.0 for every submission --
    # including the oracle, which is meant to score 1.0.
    base_src = HERE / "fixtures" / "base_model_chemberta"
    if not (base_src / "config.json").is_file():
        raise SystemExit(
            f"REFUSING: no ChemBERTa base fixture at {base_src}. "
            "Run `python research/make_chem_fixture.py` first."
        )
    dest = grader / "base_model"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(base_src, dest)
    if not any(dest.glob("*.safetensors")) and not (dest / "pytorch_model.bin").is_file():
        raise SystemExit(f"REFUSING: base fixture at {dest} has no weights")

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
