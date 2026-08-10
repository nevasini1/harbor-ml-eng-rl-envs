"""Materialize the ChemBERTa base fixture the mol-property verifier needs.

assemble_task.py copies research/fixtures/base_model_chemberta into the verifier
build context as /grader/base_model, where grade.py reads it for the lineage
check. That directory has never existed -- make_fixtures.py builds ESM fixtures
for the protein task only -- and assemble_task.py skipped the copy silently when
it was missing. The result: tests/Dockerfile's `COPY grader/ /grader/` fails, and
if it did build, read_config would raise on the absent /grader/base_model, every
eval set would floor, and the reward would be 0.0 for every submission including
the oracle.

The revision is pinned to the same commit the agent environment bakes
(tasks/mol-property-adapt/environment/Dockerfile:41). The lineage check compares
the submission tensor-by-tensor against this copy, so it must be the exact bytes
the agent started from -- a different revision would make honest fine-tunes look
like substitutions.

Run:  python research/make_chem_fixture.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

BASE_MODEL = "DeepChem/ChemBERTa-77M-MLM"
# Must match tasks/mol-property-adapt/environment/Dockerfile ARG BASE_REVISION.
BASE_REVISION = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"

HERE = Path(__file__).resolve().parent
DEST = HERE / "fixtures" / "base_model_chemberta"

# grade.py needs the config for the architecture check, weights for the cosine
# check, and the tokenizer to featurize SMILES.
REQUIRED = ("config.json", "tokenizer_config.json")
WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def main() -> int:
    from huggingface_hub import snapshot_download

    if (DEST / "config.json").is_file():
        print(f"fixture already present at {DEST}")
    else:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        tmp = DEST.with_suffix(".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"downloading {BASE_MODEL}@{BASE_REVISION[:12]} ...")
        snapshot_download(
            BASE_MODEL,
            revision=BASE_REVISION,
            local_dir=str(tmp),
            local_dir_use_symlinks=False,
            # Skip the pickled duplicate when safetensors is present; keep both
            # patterns so the check below can accept either serialization.
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx", ".git*"],
        )
        shutil.rmtree(DEST, ignore_errors=True)
        tmp.rename(DEST)

    missing = [f for f in REQUIRED if not (DEST / f).is_file()]
    if missing:
        print(f"REFUSING: fixture missing {missing}", file=sys.stderr)
        return 1
    if not any((DEST / w).is_file() for w in WEIGHTS):
        print(f"REFUSING: fixture has no weights ({' or '.join(WEIGHTS)})", file=sys.stderr)
        return 1

    cfg = json.loads((DEST / "config.json").read_text())
    print(f"\nfixture ready: {DEST}")
    print(f"  model_type={cfg.get('model_type')} hidden={cfg.get('hidden_size')} "
          f"layers={cfg.get('num_hidden_layers')}")
    for p in sorted(DEST.iterdir()):
        if p.is_file():
            print(f"  {p.name}  {p.stat().st_size:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
