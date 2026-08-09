"""Run the verifier against every fixture and check the outcome ordering.

The requirement is not that every case scores well - it is that every case
produces a valid reward and that legitimate submissions are never rejected while
substituted ones always are.
"""

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
TEST = HERE / "cache" / "crash_test.csv"

# expectation: (must_produce_reward, expected_status)
EXPECTED = {
    "base": ("ok", "legitimate: untouched encoder + fresh head"),
    "constant": ("ok", "legitimate shape, no signal -> reward floor"),
    "swap-35M": ("rejected", "architecture layer must catch the swap"),
    "shuffled": ("rejected", "lineage layer must catch same-shape wrong-lineage"),
    "corrupt": ("error|rejected", "must not raise out of the verifier"),
    "empty": ("rejected", "no config.json"),
    "no-config": ("rejected", "no config.json"),
    "hostile-config": ("rejected", "unparseable / oversized config"),
    "symlink": ("rejected", "symlinks are refused before reading"),
}


def make_test_csv(n: int = 300) -> None:
    """Small slice used only to exercise the scoring path end to end."""
    if TEST.exists():
        return
    src = HERE / "data" / "flip2" / "hydro" / "random_split.csv.gz"
    with gzip.open(src, "rt") as fh:
        df = pd.read_csv(fh)
    TEST.parent.mkdir(exist_ok=True)
    df[df["set"] == "test"].head(n)[["sequence", "target"]].to_csv(TEST, index=False)


def run(name: str) -> dict:
    out = HERE / "results" / f"crash_{name}.json"
    cmd = [
        sys.executable, str(HERE / "grade.py"),
        "--submission", str(FIX / name),
        "--base", str(FIX / "base_model"),
        "--test", str(TEST),
        "--out", str(out),
        "--public-hashes", str(HERE / "results" / "model_pins.json"),
        "--base-spearman", "0.10",
        "--ref-spearman", "0.60",
        "--max-len", "128",
        "--batch-size", "16",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    report = json.loads(out.read_text()) if out.exists() else {}
    return {"exit_code": proc.returncode, "report": report,
            "stderr_tail": proc.stderr.strip()[-300:]}


def main() -> None:
    make_test_csv()
    (HERE / "results").mkdir(exist_ok=True)
    rows, ok = [], True

    for name, (expected, why) in EXPECTED.items():
        res = run(name)
        rep = res["report"]
        status = rep.get("status", "MISSING")
        reward = rep.get("reward", None)
        passed = (status in expected.split("|")) and (reward is not None) and res["exit_code"] == 0
        ok &= passed
        rows.append({"case": name, "status": status, "reward": reward,
                     "exit": res["exit_code"], "expected": expected,
                     "pass": passed, "reason": rep.get("reason", "")[:90]})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name:<16} status={status:<9} reward={reward} "
              f"exit={res['exit_code']}")
        if rep.get("reason"):
            print(f"         reason: {rep['reason'][:150]}")
        if not passed and res["stderr_tail"]:
            print(f"         stderr: {res['stderr_tail'][:200]}")

    dest = HERE / "results" / "crash_matrix.json"
    dest.write_text(json.dumps(rows, indent=2))
    print(f"\n{'ALL CASES PRODUCED A VALID REWARD' if ok else 'SOME CASES FAILED'}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
