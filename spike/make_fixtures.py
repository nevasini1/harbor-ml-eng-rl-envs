"""Build the calibration/adversarial submission matrix for the verifier.

Cases, and what each is meant to prove:
  base           - untouched base + fresh head. Must PASS integrity, score at the floor.
  head-only      - encoder frozen, head trained. Must PASS: this is a legitimate strategy.
  finetuned      - encoder moved by a short fine-tune. Must PASS.
  swap-35M       - ESM-2-35M substituted. Must be REJECTED by the architecture layer.
  public-verbatim- the base repo's own public file re-uploaded as a "result".
                   Must PASS (it IS the provided base) - guards against a false positive.
  corrupt        - truncated safetensors. Must yield a reward, not an exception.
  empty          - empty directory.
  no-config      - weights without config.json.
  hostile-config - deeply nested JSON intended to blow the parser.
  symlink        - config.json symlinked to /dev/zero.
  constant       - head zeroed so every prediction is identical.
  shuffled       - base weights randomly permuted within each tensor. Must be REJECTED
                   by the lineage layer (same architecture, same shapes, wrong lineage).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, EsmForSequenceClassification

BASE = "facebook/esm2_t6_8M_UR50D"
BASE_REV = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
SWAP = "facebook/esm2_t12_35M_UR50D"
OUT = Path(__file__).parent / "fixtures"


def save_seqcls(repo: str, dest: Path, revision: str | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    model = EsmForSequenceClassification.from_pretrained(repo, num_labels=1, revision=revision)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    model.save_pretrained(dest)
    tok.save_pretrained(dest)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)

    print("building base ...")
    base = OUT / "base"
    save_seqcls(BASE, base, BASE_REV)

    print("building reference base_model copy (verifier side) ...")
    shutil.rmtree(OUT / "base_model", ignore_errors=True)
    shutil.copytree(base, OUT / "base_model")

    print("building swap-35M ...")
    save_seqcls(SWAP, OUT / "swap-35M")

    print("building constant ...")
    const = OUT / "constant"
    shutil.copytree(base, const, dirs_exist_ok=True)
    tensors = load_file(const / "model.safetensors")
    for key in tensors:
        if key.startswith("classifier"):
            tensors[key] = torch.zeros_like(tensors[key])
    save_file(tensors, const / "model.safetensors", metadata={"format": "pt"})

    print("building shuffled ...")
    shuf = OUT / "shuffled"
    shutil.copytree(base, shuf, dirs_exist_ok=True)
    tensors = load_file(shuf / "model.safetensors")
    gen = torch.Generator().manual_seed(0)
    for key, val in tensors.items():
        flat = val.reshape(-1)
        tensors[key] = flat[torch.randperm(flat.numel(), generator=gen)].reshape(val.shape)
    save_file(tensors, shuf / "model.safetensors", metadata={"format": "pt"})

    print("building corrupt ...")
    corrupt = OUT / "corrupt"
    shutil.copytree(base, corrupt, dirs_exist_ok=True)
    raw = (corrupt / "model.safetensors").read_bytes()
    (corrupt / "model.safetensors").write_bytes(raw[: len(raw) // 3])

    print("building empty / no-config / hostile-config / symlink ...")
    (OUT / "empty").mkdir(exist_ok=True)

    noconf = OUT / "no-config"
    shutil.copytree(base, noconf, dirs_exist_ok=True)
    (noconf / "config.json").unlink()

    hostile = OUT / "hostile-config"
    shutil.copytree(base, hostile, dirs_exist_ok=True)
    (hostile / "config.json").write_text("[" * 100_000 + "]" * 100_000)

    sym = OUT / "symlink"
    shutil.copytree(base, sym, dirs_exist_ok=True)
    (sym / "config.json").unlink()
    (sym / "config.json").symlink_to("/dev/zero")

    print(f"\nfixtures in {OUT}:")
    for p in sorted(OUT.iterdir()):
        n = sum(1 for _ in p.rglob("*")) if p.is_dir() else 0
        print(f"  {p.name:<18} {n} entries")


if __name__ == "__main__":
    main()
