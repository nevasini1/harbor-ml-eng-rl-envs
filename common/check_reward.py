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
# hand-assembled from the spike results.
#
# The protein task has no entry because it ships `tiers.json`, which states
# thresholds rather than anchors, so there is nothing here to compare a shipped
# base/reference against. That is a real hole and it is NOT silently accepted:
# `MEASURED_BANDS` below covers its measurement record, and `shipping.report()`
# prints a NOT EXAMINED line for any task with no anchors.
#
# (The comment that used to sit here said protein's absence was "itself recorded
# below rather than skipped" -- it was not, it was skipped -- and that protein has
# "no upstream ladder at all" -- it has the best one in the repo, `lpft.json`, the
# only record with per-seed rows. Both claims were false, in the checker written to
# catch exactly that.)
UPSTREAM = {
    "qa-sft-adapt": ROOT / "research/posttrain/results/qa_anchors.json",
    "pref-reward-model": ROOT / "research/posttrain/results/rm_anchors.json",
    "mol-property-adapt": ROOT / "research/results/anchors_private.json",
}

# Measurement records that state a `band_sigma` next to the per-arm noise it was
# computed from, and so can be checked for self-consistency against
# common/shipping.py regardless of whether the task ships anchors. This is how the
# protein track gets covered at all.
MEASURED_BANDS = (
    ("sciml-protein-regression/meltome",
     ROOT / "tasks/sciml-protein-regression/scripts/lpft.json", "frozen", "lpft"),
)

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
            return (_std(arms[base_arm], name, f"{base_arm}/{name}"),
                    _std(arms[ref_arm], name, f"{ref_arm}/{name}"))
        return None
    measured = doc.get(name, {}).get("measured_arms_n5_legal")   # hand-assembled mol
    if not measured:
        return None
    metric = _metric_of(anchor)
    base_arm = min(measured, key=lambda k: abs(measured[k]["mean"] - anchor[f"base_{metric}"]))
    ref_arm = max(measured, key=lambda k: measured[k]["mean"])
    return _std(measured, base_arm, name), _std(measured, ref_arm, name)


def _std(arms: dict, arm: str, where: str) -> float:
    """The arm's seed spread, or a refusal.

    Not `.get("std", 0.0)`. `shipping.evaluate` divides the band by this, so a
    missing value used to become 0.0 and -- when both arms were missing -- floor
    `sigma` to 1e-9, making `band_sigma` about 1e7 and the rerun noise ~0. A
    malformed measurement record passed the criterion by seven orders of magnitude
    and read as the most precise eval set in the repo. A deterministic arm records
    `std: 0.0` explicitly and is fine; an arm with no `std` key at all is not a
    measurement.
    """
    if "std" not in arms[arm]:
        raise ValueError(
            f"arm {arm!r} for {where} records no `std`; band_sigma divides by the "
            "arms' seed spread, so this cannot be defaulted")
    return float(arms[arm]["std"])


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

    # Measurement records that state a band_sigma without shipping anchors. Without
    # this loop the protein track was covered by nothing at all, and `lpft.json`
    # kept a quadrature-derived 3.92 for a full day after the convention changed --
    # found by hand, which is what this file exists to make unnecessary.
    for label, path, base_key, ref_key in MEASURED_BANDS:
        if not path.exists():
            fails.append(f"sigma-convention: {label} record missing at "
                         f"{path.relative_to(ROOT)}")
            continue
        doc = json.loads(path.read_text())
        if "band_sigma" not in doc or "band" not in doc:
            continue
        b_std = _std(doc, base_key, f"{label}/{base_key}")
        r_std = _std(doc, ref_key, f"{label}/{ref_key}")
        expected = doc["band"] / max(b_std, r_std, 1e-9)
        if abs(expected - doc["band_sigma"]) > 0.05:
            fails.append(
                f"sigma-convention: {label} reports band_sigma "
                f"{doc['band_sigma']:.2f} in {path.relative_to(ROOT)}, but the "
                f"shared rule gives {expected:.2f} from its own recorded arm noise "
                f"({b_std}, {r_std}).")


def check_no_retired_criterion(fails: list[str]) -> None:
    """A measurement record must state one bar, not two.

    `finalize_anchors.py` writes its output with `data.update(out)`, which merges
    into whatever the file already held. `criterion_record()` writes the key
    `criterion`; an earlier version of the pipeline wrote `criteria`. Merging
    never removes, so both survive -- one character apart, undated, with nothing
    marking which is authoritative:

        "criteria":  {"min_band_sigma": 3.0, "min_band": 0.02}   <- retired
        "criterion": {"min_band_sigma": 4.0, ...}                 <- current

    `min_band` was the second test that `common/shipping.py`'s docstring records as
    "removed after it excluded the eval set I wanted to keep". It sat in the qa
    record at 0.02 and in the rm record at 0.0, read by nothing, which is exactly
    why nobody noticed.

    STATUS: both keys were removed and `finalize_anchors.py` now clears the derived
    keys before re-merging, so this check is a REGRESSION GUARD -- it currently
    finds nothing and is expected to. The paragraph above is written in the past
    tense on purpose: an earlier version of this docstring described the problem in
    the present tense after it had been fixed, which makes a check's own
    documentation lie about the repo. If this ever fires again, the fix is to clear
    the stale key at the write site rather than here.
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
    """Every measured number a figure could want, and the file that owns it.

    Covers the shipped anchors AND the upstream measurement records. The upstream
    half was missing, which is why seven random-init means typed into
    `plot_criterion.py` went unnoticed: they live under `arms.random_init.*.mean`
    in the ladder JSONs and nothing here claimed them.
    """
    owned: dict[float, str] = {}

    def claim(value: object, rel: str) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            owned.setdefault(round(float(value), 6), rel)

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
                if k.startswith(("base_", "reference_", "t_")) or k == "band":
                    claim(v, rel)

    # Upstream ladders: every arm's mean and std, on every eval set.
    for path in list(UPSTREAM.values()) + [p for _, p, _, _ in MEASURED_BANDS]:
        if not path.exists():
            continue
        rel = str(path.relative_to(ROOT))
        doc = json.loads(path.read_text())
        for arms in ([doc.get("arms", {})] +
                     [v.get("measured_arms_n5_legal", {})
                      for v in doc.values() if isinstance(v, dict)]):
            for arm in arms.values():
                if not isinstance(arm, dict):
                    continue
                for k in ("mean", "std"):
                    claim(arm.get(k), rel)
                for sub in arm.values():          # arms[arm][eval_set][mean|std]
                    if isinstance(sub, dict):
                        claim(sub.get("mean"), rel)
                        claim(sub.get("std"), rel)
        for k in ("band", "band_sigma"):
            claim(doc.get(k), rel)
        for k in ("frozen", "lpft"):
            if isinstance(doc.get(k), dict):
                claim(doc[k].get("mean"), rel)
    return owned


def check_figures_read_their_data(fails: list[str]) -> None:
    """A figure that hardcodes an anchor renders a claim nobody re-checks.

    Scope: module-level constant blocks only -- a line starting an ALL_CAPS
    assignment, and its continuation lines. That is where data gets typed in
    (`RANDOM_INIT = {...}`, the old `PANELS = [...]`), and it excludes the layout
    floats that live in call arguments like `ax.text(0.982, 0.025, ...)`.

    This replaces a much weaker rule that skipped any script containing
    `json.load` or `read_text` at all. Every plot script reads *something*, so
    that version examined zero scripts and its `ok` was vacuous -- it missed seven
    measured random-init means typed into `plot_criterion.py`, under a docstring
    saying "read from the committed anchor files, not typed in", in the script the
    README cites as the counter-example to hardcoding.
    """
    owned = _committed_values()
    for script in PLOT_SCRIPTS:
        lines, in_block, block = script.read_text().splitlines(), False, []
        for line in lines:
            if re.match(r"^[A-Z][A-Z0-9_]*\s*=", line):
                in_block = True
            elif in_block and line and not line[0].isspace() and not line.startswith(
                    (")", "]", "}")):
                in_block = False
            if in_block:
                block.append(line)
        src = "\n".join(block)
        literals = {round(float(m), 6) for m in re.findall(r"\b\d+\.\d{2,}\b", src)}
        hits = sorted(literals & owned.keys())
        if hits:
            where = ", ".join(f"{v} (owned by {owned[v]})" for v in hits[:4])
            fails.append(
                f"figures-read-their-data: {script.relative_to(ROOT)} has {len(hits)} "
                f"measured value(s) typed into a module-level constant: {where}"
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
KNOWN: dict[str, str] = {}


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
