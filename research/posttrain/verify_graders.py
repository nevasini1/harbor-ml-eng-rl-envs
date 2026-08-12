"""Regression-test both post-training verifiers against the real images.

The protein task has `modal_verify_hardening.py`; this is the same idea for the
two post-training tasks, run through local Docker with `--network none`, which is
what the production verifier is.

It asserts the **accept** path first. A grader that rejects honest submissions is
worse than the loopholes it closes: a false reject is indistinguishable, in the
reward, from an agent that did nothing, and it silently deletes the signal the
whole environment exists to produce.

Fixtures, built here rather than committed, because a checkpoint is 300 MB:

  accept   base_unchanged   the provided base with a fresh head. Legitimate and
                            lazy: must be accepted and must score ~0 recovery.
  accept   oracle           the reference checkpoint, if it has been trained.
  reject   shuffled         one encoder tensor's values permuted. Same shapes,
                            same names, same norms -- only the per-tensor cosine
                            sees it.
  reject   nan              one encoder tensor NaN'd. Tests that non-finite
                            weights are rejected rather than skipped, which is
                            the bug that let an attacker NaN exactly the tensors
                            that would have scored badly.
  reject   public_twin      a bit-identical copy of a *different* public
                            checkpoint with the same architecture. Only the
                            sha256 layer catches this one.
  reject   laundered        (qa only) the same-architecture instruction-tuned
                            sibling, lightly perturbed so it is not bit-identical.
                            Passes arch, sha and cosine; only the nearest-ancestor
                            check separates it from an honest fine-tune.
  reject   truncated        config.json deleted. Malformed input must floor to 0,
                            not crash the grader.
  reject   contaminated     a training log containing held-out text.

Every case must produce a reward.json. That is the invariant that matters most:
a missing reward file is a trial error, not a zero.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
RESULTS = HERE / "results"

TRACKS = {
    "rm": {
        "task": ROOT / "tasks" / "pref-reward-model",
        "image": "pref-reward-model-verifier:verify",
        "num_labels": 1,
        "kind": "seq",
        "text_cols": ("prompt", "chosen"),
        # Same architecture as distilroberta-base, different weights, and its
        # sha256 is in public_hashes.json.
        "public_twin": "sentence-transformers/all-distilroberta-v1",
    },
    "qa": {
        "task": ROOT / "tasks" / "qa-sft-adapt",
        "image": "qa-sft-adapt-verifier:verify",
        "kind": "causal",
        "text_cols": ("question",),
    },
}


def build_image(cfg: dict) -> None:
    print(f"==> building {cfg['image']}")
    subprocess.run(["docker", "build", "-q", "-t", cfg["image"],
                    str(cfg["task"] / "tests")], check=True,
                   stdout=subprocess.DEVNULL)


def load_base(cfg: dict):
    from transformers import (AutoModelForCausalLM,
                              AutoModelForSequenceClassification, AutoTokenizer)

    base = cfg["task"] / "tests" / "grader" / "base_model"
    tok = AutoTokenizer.from_pretrained(str(base))
    if cfg["kind"] == "seq":
        model = AutoModelForSequenceClassification.from_pretrained(
            str(base), num_labels=cfg["num_labels"])
    else:
        model = AutoModelForCausalLM.from_pretrained(str(base))
    return model, tok


def save(model, tok, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tok.save_pretrained(dest)
    return dest


def make_fixtures(cfg: dict, work: Path) -> dict:
    import torch

    fixtures: dict[str, Path] = {}
    model, tok = load_base(cfg)
    fixtures["base_unchanged"] = save(model, tok, work / "base_unchanged")

    # shuffled: permute the values of one encoder tensor. Shapes, names and norms
    # all survive; only a per-tensor cosine notices.
    model2, tok2 = load_base(cfg)
    with torch.no_grad():
        for name, p in model2.named_parameters():
            if p.dim() == 2 and "embed" in name:
                flat = p.view(-1)
                p.copy_(flat[torch.randperm(flat.numel())].view(p.shape))
                break
    fixtures["shuffled"] = save(model2, tok2, work / "shuffled")

    model3, tok3 = load_base(cfg)
    with torch.no_grad():
        for name, p in model3.named_parameters():
            if p.dim() == 2 and "embed" in name:
                p[0].fill_(float("nan"))
                break
    fixtures["nan"] = save(model3, tok3, work / "nan")

    truncated = work / "truncated"
    shutil.copytree(fixtures["base_unchanged"], truncated, dirs_exist_ok=True)
    (truncated / "config.json").unlink()
    fixtures["truncated"] = truncated

    oracle = HERE / "fixtures" / ("rm_oracle" if cfg["kind"] == "seq" else "qa_oracle")
    if (oracle / "config.json").exists():
        fixtures["oracle"] = oracle

    # public_twin: a bit-identical copy of a *different* public checkpoint that
    # happens to share the base's architecture, so layers 1 and 3 both pass it.
    # Only the sha256 layer sees it. Assembled from the twin's weights and the
    # base's tokenizer, which is what an agent swapping in a stronger encoder
    # would actually ship.
    twin_repo = cfg.get("public_twin")
    if twin_repo:
        from huggingface_hub import snapshot_download

        twin = work / "public_twin"
        snapshot_download(twin_repo, local_dir=str(twin),
                          allow_patterns=["config.json", "model.safetensors"])
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "special_tokens_map.json"):
            src = cfg["task"] / "tests" / "grader" / "base_model" / f
            if src.exists():
                shutil.copy2(src, twin / f)
        fixtures["public_twin"] = twin

    laundered = cfg["task"] / "tests" / "grader" / "siblings"
    if laundered.is_dir():
        sib = next((p for p in sorted(laundered.iterdir()) if p.is_dir()), None)
        if sib is not None:
            from transformers import AutoModelForCausalLM

            m = AutoModelForCausalLM.from_pretrained(str(sib))
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * 1e-4)
                    break
            fixtures["laundered"] = save(m, tok, work / "laundered")

    return fixtures


def run_case(cfg: dict, sub: Path, logs: Path | None, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mounts = ["-v", f"{sub.resolve()}:/app/final_model:ro"]
    if logs is not None:
        mounts += ["-v", f"{logs.resolve()}:/logs/agent:ro"]
    subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--memory", "8g",
         *mounts, "-v", f"{out.resolve()}:/logs/verifier", cfg["image"],
         "bash", "/tests/test.sh"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    reward = json.loads((out / "reward.json").read_text()) \
        if (out / "reward.json").exists() else None
    metrics = json.loads((out / "metrics.json").read_text()) \
        if (out / "metrics.json").exists() else {}
    return {"reward": reward, "metrics": metrics}


def contaminated_log(cfg: dict, work: Path) -> Path:
    """A train_log.txt quoting held-out text, the way a leak actually looks.

    The eval set is read from the shipped anchors rather than named here. Which
    eval sets ship is a measurement outcome -- the preference track screened four
    and ships one -- so a hardcoded filename goes stale the moment a ladder is
    re-measured, and takes the contamination assertion down with it.
    """
    import pandas as pd

    priv_dir = cfg["task"] / "tests" / "grader" / "private"
    shipped = sorted(json.loads((priv_dir / "anchors.json").read_text()))
    if not shipped:
        raise SystemExit(f"no shipped eval sets in {priv_dir}/anchors.json")
    df = pd.read_csv(priv_dir / f"{shipped[0]}_test.csv").head(5)
    body = ["tried a few things, best val so far 0.61", "sample rows I trained on:"]
    for _, r in df.iterrows():
        for col in cfg["text_cols"]:
            body.append(str(r[col]))
    d = work / "contaminated_logs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train_log.txt").write_text("\n".join(body))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["rm", "qa", "both"], default="both")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--keep", default="", help="keep fixtures in this directory")
    args = ap.parse_args()

    failures = []
    recorded: dict[str, dict] = {}
    for track, cfg in TRACKS.items():
        if args.track not in (track, "both"):
            continue
        if not (cfg["task"] / "tests" / "grader" / "private" / "anchors.json").exists():
            print(f"SKIP {track}: task not assembled yet")
            continue
        if not args.no_build:
            build_image(cfg)

        work = Path(args.keep) / track if args.keep else \
            Path(tempfile.mkdtemp(prefix=f"verify-{track}-"))
        work.mkdir(parents=True, exist_ok=True)
        print(f"==> {track}: fixtures in {work}")
        fixtures = make_fixtures(cfg, work)

        # Accept path first, deliberately.
        order = [n for n in ("base_unchanged", "oracle") if n in fixtures]
        order += [n for n in fixtures if n not in order]
        expect_accept = {"base_unchanged", "oracle"}

        clean_logs = work / "clean_logs"
        clean_logs.mkdir(exist_ok=True)
        (clean_logs / "train_log.txt").write_text("trained a reward model\n")

        for name in order:
            res = run_case(cfg, fixtures[name], clean_logs, work / f"out_{name}")
            if res["reward"] is None:
                failures.append(f"{track}/{name}: NO reward.json written")
                print(f"  {name:<16} !! no reward.json")
                continue
            sets = res["metrics"].get("eval_sets", {})
            statuses = {k: v.get("status") for k, v in sets.items()}
            ok = all(s == "ok" for s in statuses.values()) and statuses
            reward = res["reward"]["reward"]
            verdict = "accept" if ok else "reject"
            want = "accept" if name in expect_accept else "reject"
            mark = "ok " if verdict == want else "FAIL"
            if verdict != want:
                reason = next((v.get("reason") for v in sets.values()
                               if v.get("reason")), res["metrics"].get("reason"))
                failures.append(f"{track}/{name}: expected {want}, got {verdict}"
                                f" ({str(reason)[:120]})")
            detail = ""
            if sets:
                detail = "  ".join(
                    f"{k}={v.get('acc')}/{v.get('recovery')}" for k, v in sets.items())
            print(f"  {mark} {name:<16} reward={reward:<10} {verdict:<7} {detail}")
            # Recorded, not just printed. These results are the §4 calibration
            # evidence -- "report the reward each produced and confirm the ordering
            # matches expectation" -- and for a long time the only committed trace
            # of them was a hand-written table in RESULTS.md, which no tool could
            # check and nothing kept in step with a re-run.
            recorded.setdefault(cfg["task"].name, {})[name] = {
                "verdict": verdict, "reward": reward,
                "eval_sets": {k: {"metric": v.get("acc"),
                                  "recovery": v.get("recovery"),
                                  "status": v.get("status")}
                              for k, v in sets.items()},
            }
            if name in expect_accept and reward is not None:
                metric_vals = [v.get("acc") for v in sets.values() if v.get("acc")]
                if name == "base_unchanged" and reward > 0.25:
                    failures.append(
                        f"{track}/base_unchanged: reward {reward} -- the base "
                        "anchor is too low, a no-op submission is being paid")
                del metric_vals

        # Contamination: same accepted checkpoint, a log that quotes held-out text.
        bad_logs = contaminated_log(cfg, work)
        res = run_case(cfg, fixtures["base_unchanged"], bad_logs,
                       work / "out_contaminated")
        sets = res["metrics"].get("eval_sets", {})
        rejected = [k for k, v in sets.items() if v.get("status") == "rejected"]
        if not rejected:
            failures.append(f"{track}/contaminated: held-out text in the agent log "
                            "was not detected")
        print(f"  {'ok ' if rejected else 'FAIL'} contaminated     "
              f"rejected={rejected or 'none'}")

    out = RESULTS / "verifier_fixtures.json"
    out.write_text(json.dumps({
        "_provenance": {
            "source": "research/posttrain/verify_graders.py",
            "measured": True,
            "regenerate_with": "python research/posttrain/verify_graders.py",
        },
        "tracks": recorded,
    }, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)} "
          f"({sum(len(v) for v in recorded.values())} fixture results)")

    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all verifier assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
