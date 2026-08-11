"""Assert that the reward's story is the same everywhere it is told.

Why this exists
---------------
The reward is derived carefully and recorded in several places: a measurement
record under `research/`, a shipped `anchors.json` inside each verifier image, a
task README, a `task.toml`, and a figure. Nothing checked that those places still
agree, and they stopped agreeing.

`research/plot_ladder.py` hardcoded its numbers instead of reading them, so
`research/results/anchor_ladder.png` still captions the protein panel "Inverted --
unfixable" -- a verdict `tasks/sciml-protein-regression/README.md` overturned two
days later. That figure is embedded in a *different* task's README, so one shipped
task's docs asserted a conclusion another shipped task's docs called wrong.

This is the same shape of problem `common/sync.py --check` already solves for the
grader modules: copies drift silently, so make drift loud. That script is the
model for this one.

What it checks
--------------
  anchors-match-upstream  every shipped anchors.json still agrees with the
                          measurement record it was derived from. This is the one
                          that matters most: the shipped file is what divides.

  sigma-convention        `band_sigma` means the same thing in every task that
                          reports one. It currently does not, and the READMEs
                          compare the two tracks' figures as though it does.

  criterion-stamped       every shipped anchor records the bar it passed, so an
                          anchor measured under the old `band_sigma >= 3.0` rule
                          is distinguishable from one measured under 4.0.

  reward-declared         every task.toml declares `reward_definition`, so the
                          reward shape is readable without opening the grader.

  figures-read-their-data no plotting script hardcodes a value that lives in a
                          committed anchor or tier file. Reading the file is the
                          only way a figure stays true after a re-measurement.

A failure here is not necessarily a bug in the reward. It is a claim that two
files disagree, printed with both sides, so the disagreement gets resolved on
purpose rather than discovered in a figure two days later.

    python common/check_reward.py
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"

# Shipped anchor file -> the measurement record it was derived from. The two
# post-training tracks come from `finalize_anchors.py`; mol predates it and was
# hand-assembled from the spike results. Protein ships tiers with no upstream
# ladder at all, which is itself recorded below rather than skipped.
UPSTREAM = {
    "qa-sft-adapt": ROOT / "research/posttrain/results/qa_anchors.json",
    "pref-reward-model": ROOT / "research/posttrain/results/rm_anchors.json",
    "mol-property-adapt": ROOT / "research/results/anchors_private.json",
}

METRICS = ("auc", "acc", "spearman")

PLOT_SCRIPTS = sorted((ROOT / "research").glob("plot_*.py"))


def _metric_of(anchor: dict) -> str | None:
    return next((m for m in METRICS if f"base_{m}" in anchor), None)


def _shipped(task: str) -> tuple[Path, dict]:
    """The file a verifier image actually divides by."""
    p = TASKS / task / "tests/grader/private/anchors.json"
    if p.exists():
        return p, json.loads(p.read_text())
    p = TASKS / task / "tests/tiers.json"
    if p.exists():
        return p, json.loads(p.read_text())
    return p, {}


def check_anchors_match_upstream(fails: list[str]) -> None:
    for task, up_path in sorted(UPSTREAM.items()):
        ship_path, shipped = _shipped(task)
        if not shipped:
            fails.append(f"anchors-match-upstream: {task} has no shipped anchors at "
                         f"{ship_path.relative_to(ROOT)}")
            continue
        if not up_path.exists():
            fails.append(f"anchors-match-upstream: {task} shipped anchors have no "
                         f"upstream record at {up_path.relative_to(ROOT)}")
            continue
        doc = json.loads(up_path.read_text())
        # finalize_anchors.py nests under "anchors"; the hand-assembled mol record
        # is a bare mapping of eval set -> anchor.
        upstream = doc.get("anchors") or doc.get("screened") or doc
        for name, anc in shipped.items():
            metric = _metric_of(anc)
            if metric is None:
                continue
            src = upstream.get(name)
            if not isinstance(src, dict):
                fails.append(f"anchors-match-upstream: {task}/{name} is shipped but "
                             f"absent from {up_path.relative_to(ROOT)}")
                continue
            for field in (f"base_{metric}", f"reference_{metric}"):
                if field not in src:
                    continue
                if abs(float(anc[field]) - float(src[field])) > 1e-9:
                    fails.append(
                        f"anchors-match-upstream: {task}/{name}.{field} is "
                        f"{anc[field]} in {ship_path.relative_to(ROOT)} but "
                        f"{src[field]} in {up_path.relative_to(ROOT)}")


def check_criterion_stamped(fails: list[str]) -> None:
    """An anchor should say which bar it cleared.

    Without this an anchor measured under the retired `band_sigma >= 3.0` rule is
    indistinguishable from one measured under the derived 4.0 bar, and the only
    way to tell is to date the commit that wrote it.
    """
    for task in sorted(d.name for d in TASKS.iterdir() if d.is_dir()):
        ship_path, shipped = _shipped(task)
        if not shipped or ship_path.name == "tiers.json":
            continue
        for name, anc in shipped.items():
            if not isinstance(anc, dict) or _metric_of(anc) is None:
                continue
            if "min_band_sigma" not in anc:
                fails.append(
                    f"criterion-stamped: {task}/{name} records no min_band_sigma, "
                    f"so which bar it passed is not recoverable from "
                    f"{ship_path.relative_to(ROOT)}")


def _arm_stds(name: str, anchor: dict, doc: dict) -> tuple[float, float] | None:
    """(base_std, reference_std) for the two arms that define this band."""
    if "arms" in doc:                                   # finalize_anchors.py record
        arms, base_arm = doc["arms"], anchor.get("base_arm")
        ref_arm = next((a for a in ("sft_full", "finetune") if a in arms), None)
        if base_arm in arms and ref_arm and name in arms[base_arm]:
            return arms[base_arm][name]["std"], arms[ref_arm][name]["std"]
        return None
    measured = doc.get(name, {}).get("measured_arms_n5_legal")   # hand-assembled mol
    if not measured:
        return None
    metric = _metric_of(anchor)
    base_arm = min(measured, key=lambda k: abs(measured[k]["mean"] - anchor[f"base_{metric}"]))
    ref_arm = max(measured, key=lambda k: measured[k]["mean"])
    return measured[base_arm].get("std", 0.0), measured[ref_arm].get("std", 0.0)


def check_sigma_convention(fails: list[str]) -> None:
    """`band_sigma` must mean the same thing in every task that reports one.

    `common/shipping.py` defines it as `band / max(base_std, ref_std)` -- the
    larger of the two arms, "so a noisy base cannot be averaged away by a tight
    reference". The mol anchors were computed before that rule existed and use
    `band / sqrt(base_std**2 + ref_std**2)` instead, over the frozen *head* even on
    bbbp, where the shipped base is the deterministic logistic probe.

    Nothing ships or fails differently -- quadrature is the more conservative of
    the two, and both mol eval sets clear 4.0 either way. What breaks is
    comparison: the READMEs quote bbbp's 4.09 sigma repo-wide as the tightest band
    that ships, and under the shared rule it is 6.81. A sigma from one task cannot
    be held up against a sigma from another until they are the same quantity.
    """
    for task, up_path in sorted(UPSTREAM.items()):
        _, shipped = _shipped(task)
        if not shipped or not up_path.exists():
            continue
        doc = json.loads(up_path.read_text())
        for name, anc in shipped.items():
            if not isinstance(anc, dict) or "band_sigma" not in anc:
                continue
            stds = _arm_stds(name, anc, doc)
            if stds is None:
                continue
            expected = anc["band"] / max(*stds, 1e-9)
            if abs(expected - anc["band_sigma"]) > 0.05:
                fails.append(
                    f"sigma-convention: {task}/{name} reports band_sigma "
                    f"{anc['band_sigma']:.2f}, but common/shipping.py's rule "
                    f"(band / max(base_std, ref_std)) gives {expected:.2f} from the "
                    f"recorded arm noise {stds}. The two tracks compute it differently, "
                    "so their sigmas are not comparable.")


def check_no_retired_criterion(fails: list[str]) -> None:
    """A measurement record must state one bar, not two.

    `finalize_anchors.py` writes its output with `data.update(out)`, which merges
    into whatever the file already held. `criterion_record()` writes the key
    `criterion`; an earlier version of the pipeline wrote `criteria`. Merging
    never removes, so both survive -- one character apart, undated, with nothing
    marking which is authoritative:

        "criteria":  {"min_band_sigma": 3.0, "min_band": 0.02}   <- retired
        "criterion": {"min_band_sigma": 4.0, ...}                 <- current

    `min_band` is the second test that `common/shipping.py`'s docstring records as
    "removed after it excluded the eval set I wanted to keep". It is still sitting
    in the qa record at 0.02 and in the rm record at 0.0. Nothing reads either
    key, which is exactly why nobody noticed.

    Keeping the old bar is fine -- that is history worth having. Keeping it under
    a name that looks like the current one is not. Move it under `superseded`,
    with the date it stopped applying.
    """
    for task, up_path in sorted(UPSTREAM.items()):
        if not up_path.exists():
            continue
        doc = json.loads(up_path.read_text())
        if "criteria" in doc and "criterion" in doc:
            retired = doc["criteria"].get("min_band_sigma")
            current = doc["criterion"].get("min_band_sigma")
            fails.append(
                f"retired-criterion: {up_path.relative_to(ROOT)} states two bars — "
                f"'criteria' (retired, min_band_sigma {retired}, plus the removed "
                f"min_band {doc['criteria'].get('min_band')}) alongside 'criterion' "
                f"(current, {current}). Nothing reads the retired one; the names "
                "differ by one character.")


def check_reward_declared(fails: list[str]) -> None:
    for toml_path in sorted(TASKS.glob("*/task.toml")):
        meta = tomllib.loads(toml_path.read_text()).get("metadata", {})
        if not meta.get("reward_definition"):
            fails.append(f"reward-declared: {toml_path.relative_to(ROOT)} [metadata] "
                         "has no reward_definition")


def _committed_values() -> dict[float, str]:
    """Every number a figure could legitimately want, and the file that owns it."""
    owned: dict[float, str] = {}
    for task in sorted(d.name for d in TASKS.iterdir() if d.is_dir()):
        path, doc = _shipped(task)
        if not doc:
            continue
        rel = str(path.relative_to(ROOT))
        entries = doc.values() if path.name != "tiers.json" else [doc]
        for anc in entries:
            if not isinstance(anc, dict):
                continue
            for k, v in anc.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and (
                        k.startswith(("base_", "reference_", "t_")) or k == "band"):
                    owned.setdefault(float(v), rel)
    return owned


def check_figures_read_their_data(fails: list[str]) -> None:
    """A figure that hardcodes an anchor renders a claim nobody re-checks.

    The test is deliberately narrow: a literal only counts against a script if
    that exact value is owned by a committed anchor or tier file *and* the script
    never reads a data file. Scripts that read their inputs may of course also
    contain incidental numbers.
    """
    owned = _committed_values()
    for script in PLOT_SCRIPTS:
        src = script.read_text()
        if "json.load" in src or "read_text" in src:
            continue
        literals = {float(m) for m in re.findall(r"\b0\.\d{3,}\b", src)}
        hits = sorted(literals & owned.keys())
        if hits:
            where = ", ".join(f"{v} (owned by {owned[v]})" for v in hits[:4])
            fails.append(
                f"figures-read-their-data: {script.relative_to(ROOT)} reads no data "
                f"file but hardcodes {len(hits)} committed value(s): {where}"
                + (" ..." if len(hits) > 4 else ""))


CHECKS = (
    ("anchors-match-upstream", check_anchors_match_upstream),
    ("sigma-convention", check_sigma_convention),
    ("retired-criterion", check_no_retired_criterion),
    ("criterion-stamped", check_criterion_stamped),
    ("reward-declared", check_reward_declared),
    ("figures-read-their-data", check_figures_read_their_data),
)


# Disagreements that are acknowledged and tracked rather than fixed. Each is
# printed on every run, so nothing here is hidden -- but it does not fail the
# build, because a check that is permanently red stops being read. Anything NOT
# listed here is new drift and fails.
#
# This mirrors the `provisional` stamp on `pref-reward-model`: shipping something
# known-imperfect is allowed, on purpose, with the reason recorded next to it.
# The goal is an empty dict.
KNOWN = {
    "sigma-convention: mol-property-adapt/tox21":
        "mol's anchors predate common/shipping.py and combine the two arms' noise "
        "in quadrature rather than taking the larger. Quadrature is the more "
        "conservative of the two and the verdict is unchanged (6.48 vs 7.62, bar "
        "4.0). Resolve by re-deriving mol through the shared criterion.",
    "sigma-convention: mol-property-adapt/bbbp":
        "Same convention gap, and bbbp additionally uses the frozen *head*'s noise "
        "while its shipped base is the deterministic logistic probe: 4.09 recorded "
        "against 6.81 under the shared rule. This is the figure the READMEs quote "
        "repo-wide as the tightest band that ships, so re-deriving it moves prose "
        "in several files.",
    "criterion-stamped: mol-property-adapt/tox21":
        "Stamping the bar onto anchors computed under a different sigma convention "
        "would assert a comparability that does not hold yet. Fix with the above.",
    "criterion-stamped: mol-property-adapt/bbbp":
        "As above.",
}


def _key(message: str) -> str:
    """`check-name: subject`, the part of a finding that identifies it."""
    return " ".join(message.split()[:2])


def main() -> int:
    new: list[str] = []
    known: list[str] = []
    for label, fn in CHECKS:
        fails: list[str] = []
        fn(fails)
        fresh = [f for f in fails if _key(f) not in KNOWN]
        known += [f for f in fails if _key(f) in KNOWN]
        status = "ok  " if not fresh else "FAIL"
        note = ""
        if fails and not fresh:
            status, note = "ok  ", f"  ({len(fails)} known)"
        elif fresh:
            note = f"  ({len(fresh)} new)"
        print(f"{status}  {label}{note}")
        new += fresh

    if known:
        print("\nknown, tracked, not fixed:")
        for f in known:
            print(f"  - {f}")
            print(f"    why not fixed: {KNOWN[_key(f)]}")

    if new:
        print("\nNEW disagreements — two files making different claims about the "
              "same reward:")
        for f in new:
            print(f"  {f}")
        return 1
    print("\nno new drift" + (f"; {len(known)} known issue(s) above" if known
                              else "; the reward says the same thing everywhere"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
