"""Resolve pinned revisions and public file hashes for the base model and its
plausible substitution targets.

The sha256 values here are what a verifier compares a submitted checkpoint
against: matching any public checkpoint other than the provided base is grounds
for rejection.
"""

import json
from pathlib import Path

from huggingface_hub import HfApi

REPOS = [
    "facebook/esm2_t6_8M_UR50D",    # the provided base
    "facebook/esm2_t12_35M_UR50D",  # substitution targets below
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
]

ARCH_FIELDS = [
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "position_embedding_type",
    "emb_layer_norm_before",
    "token_dropout",
]


def main() -> None:
    api = HfApi()
    out = {}
    for repo in REPOS:
        info = api.model_info(repo, files_metadata=True)
        files = {}
        for sib in info.siblings:
            entry = {"size": sib.size}
            if sib.lfs is not None:
                entry["sha256"] = sib.lfs.sha256
            files[sib.rfilename] = entry
        out[repo] = {"sha": info.sha, "files": files}
        print(f"\n{repo}")
        print(f"  revision : {info.sha}")
        for name, meta in files.items():
            if name.endswith((".safetensors", ".bin", "config.json")):
                print(f"  {name:<28} size={meta.get('size')} sha256={meta.get('sha256')}")

    dest = Path(__file__).parent / "results" / "model_pins.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
