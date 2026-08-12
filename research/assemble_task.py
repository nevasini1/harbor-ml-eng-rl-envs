"""Populate the Harbor task tree from the spike artifacts.

Enforces the one invariant that matters: nothing under split/private and no anchor
index or seed may land in the agent image. The agent build context gets training
data only; the private region, the anchors and the base-model copy for the lineage
check go exclusively into the verifier build context.
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
TASK = HERE.parent / "tasks" / "mol-property-adapt"
EVAL_SETS = ["tox21", "bbbp"]

sys.path.insert(0, str(HERE.parent / "common"))
from shipping import criterion_record, evaluate  # noqa: E402

# The ladder behind these anchors, lowest rung first. `base` is whichever of the
# no-adaptation arms is the ceiling -- the trained head on tox21, the logistic
# probe on bbbp -- so it is recovered by matching the measured means rather than
# nominated here.
LADDER = ("frozen_logreg_cls", "frozen_head_cls", "finetune_cls")
REFERENCE_ARM = "finetune_cls"

# How many eval sets this track screened before shipping two. The selection
# correction needs it and it is not recoverable from the anchors themselves.
K_SCREENED = 5  # bbbp, tox21, clintox, bace, sider (research/SPIKE_RESULTS.md)

# The contamination tripwire's rule, matching research/posttrain/finalize_anchors.py
# so the two tracks derive it the same way. A tripwire, not proof: it flags a score
# no honest run has produced, for a human to review.
T_IMPLAUSIBLE_MARGIN = 0.15
T_IMPLAUSIBLE_FLOOR = 0.85
T_IMPLAUSIBLE_CAP = 0.98


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
    # `band` and `band_sigma` are deliberately NOT carried: they are derived below
    # through common/shipping.py, the same rule every other task is judged by. The
    # numbers in anchors_private.json predate that rule and divide the band by the
    # two arms' noise added in quadrature, where shipping.py takes the larger of
    # them -- and on bbbp the quadrature used the frozen *head*'s noise even though
    # the shipped base is the deterministic logistic probe. Deriving here means a
    # sigma from this task and a sigma from the post-training tracks are finally
    # the same quantity.
    carry = ("n_tasks", "base_auc", "reference_auc",
             "base_definition", "reference_definition")
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

        # Presence is not legality. `lock_lowdata_anchors.py` still writes this
        # same file, with the mean-pooled base and single-seed reference that were
        # measured and then deliberately discarded -- mean pooling is not
        # expressible in RobertaForSequenceClassification, so a base measured that
        # way is a ceiling over methods no agent may submit.
        #
        # Today that script is blocked only by luck: it happens not to emit
        # `t_implausible`, so the check above catches it. That is a safety property
        # held by coincidence rather than by design -- add the key to that script
        # and the off-contract anchors ship silently. So check the arm, not the
        # keys. The docstring in lock_lowdata_anchors.py says DO NOT RUN; this is
        # what makes that a refusal instead of a request.
        if "mean-pool" in m["base_definition"].lower():
            raise SystemExit(
                f"REFUSING: anchors_private.json[{name}].base_definition describes a "
                f"mean-pooled arm:\n    {m['base_definition']}\n"
                "Mean pooling is not expressible in RobertaForSequenceClassification, "
                "so that base is a ceiling over methods no legal submission can reach. "
                "Re-derive with modal_legal_anchors.py (tox21) / modal_bbbp_split.py "
                "(bbbp); do not run lock_lowdata_anchors.py, which is stale.")
        if "measured_arms_n5_legal" not in m:
            raise SystemExit(
                f"REFUSING: anchors_private.json[{name}] carries no "
                "`measured_arms_n5_legal` block, which is the record that every arm "
                "behind these anchors is a legal submission measured over 5 seeds. "
                "Without it there is nothing to distinguish a contract-legal anchor "
                "from a discarded one.")

        # Judge this eval set by the shared criterion, from the measured arms.
        arms = m["measured_arms_n5_legal"]
        base, ref = m["base_auc"], m["reference_auc"]
        base_arm = min(LADDER, key=lambda a: abs(arms[a]["mean"] - base))
        verdict = evaluate(band=ref - base,
                           base_std=arms[base_arm].get("std", 0.0),
                           ref_std=arms[REFERENCE_ARM].get("std", 0.0),
                           # The block is `measured_arms_n5_legal`; the count is in
                           # the key rather than in a field, so it is asserted here
                           # instead of silently defaulted.
                           n_seeds=5,
                           k_screened=K_SCREENED)
        if not verdict["ships"]:
            raise SystemExit(
                f"REFUSING: {name} does not pass common/shipping.py:\n  "
                + "\n  ".join(verdict["failed"])
                + "\nAn eval set that cannot carry a reward must not be assembled "
                  "into one.")

        # The contamination tripwire, by the same stated rule the post-training
        # tracks use (research/posttrain/finalize_anchors.py): a fixed margin over
        # the best score any arm has actually produced, floored so a low-ceiling
        # eval set still gets a usable bar, capped below 1.0. It was a hand-written
        # 0.85 / 0.97 in anchors_private.json -- values that happen to be right,
        # both within 0.01 of what this rule gives, but chosen rather than derived.
        # Deriving it means a re-measurement moves the tripwire with it instead of
        # leaving it pinned to whatever the ceiling was when someone last typed it.
        best_observed = max(a.get("max", a["mean"]) for a in arms.values())
        t_implausible = round(min(T_IMPLAUSIBLE_CAP,
                                  max(best_observed + T_IMPLAUSIBLE_MARGIN,
                                      T_IMPLAUSIBLE_FLOOR)), 2)
        if t_implausible <= ref:
            raise SystemExit(
                f"REFUSING: derived t_implausible {t_implausible} for {name} is not "
                f"above reference {ref}; the tripwire would fire on the recipe the "
                "reward is normalized against.")

        anchors[name] = {k: m[k] for k in carry if k in m}
        anchors[name]["t_implausible"] = t_implausible
        anchors[name]["best_observed"] = round(best_observed, 4)
        anchors[name]["base_arm"] = base_arm
        anchors[name].update({k: verdict[k] for k in (
            "band", "sigma", "band_sigma", "reward_noise_on_rerun",
            "min_band_sigma", "band_z", "z_crit_bonferroni", "k_screened",
            "n_seeds")})
        print(f"    {name}: band {verdict['band']:.4f} at "
              f"{verdict['band_sigma']:.2f} sigma (bar {verdict['min_band_sigma']}), "
              f"rerun moves the reward by {verdict['reward_noise_on_rerun']:.2f}")

    # Every top-level key must be an eval set: verifier_core.load_anchors iterates
    # this mapping and requires base_/reference_ on each value, so a sidecar
    # "_criterion" block here would fail the verifier closed. The rule travels in
    # each anchor's min_band_sigma instead, matching the post-training tasks.
    (priv / "anchors.json").write_text(json.dumps(anchors, indent=2))
    print(f"    criterion: {criterion_record()['min_band_sigma']} sigma, derived "
          f"from max_reward_noise {criterion_record()['max_reward_noise']}")

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
