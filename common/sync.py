"""Copy `common/verifier_core.py` into every task's verifier build context.

A Docker build context cannot reach outside itself, so each task's `tests/`
directory needs its own copy of the shared grader library. Copies drift silently;
this script is what makes drift loud.

    python common/sync.py            # write the copies
    python common/sync.py --check    # exit 1 if any copy differs

`--check` is the form worth wiring into CI: an edit to `common/verifier_core.py`
that never reaches `tests/` produces a verifier image that grades with the old
code while the repo shows the new code.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT / "common"

# task tests/ directory -> which shared modules its grader imports.
TARGETS = {
    ROOT / "tasks" / "mol-property-adapt" / "tests": ["verifier_core.py"],
    ROOT / "tasks" / "pref-reward-model" / "tests": ["verifier_core.py",
                                                     "textmatch.py"],
    ROOT / "tasks" / "qa-sft-adapt" / "tests": ["verifier_core.py",
                                                "textmatch.py"],
    # Partial: this grader uses the shared lineage check only. Its architecture
    # and forbidden-hash checks keep their own implementations -- different error
    # contract, and a hardcoded sha set rather than the public_hashes.json index.
    ROOT / "tasks" / "sciml-protein-regression" / "tests": ["verifier_core.py"],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    drifted, n = [], 0
    for tests_dir, names in TARGETS.items():
        for name in names:
            src, dest = COMMON / name, tests_dir / name
            want = digest(src)
            rel = dest.relative_to(ROOT)
            n += 1
            if args.check:
                if not dest.exists():
                    drifted.append(f"{rel}: missing")
                elif digest(dest) != want:
                    drifted.append(f"{rel}: differs from common/{name}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and digest(dest) == want:
                print(f"ok    {rel}")
                continue
            shutil.copy2(src, dest)
            print(f"wrote {rel}")

    if drifted:
        print("DRIFT:", file=sys.stderr)
        for d in drifted:
            print(f"  {d}", file=sys.stderr)
        return 1
    if args.check:
        print(f"all {n} copies match their source in common/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
