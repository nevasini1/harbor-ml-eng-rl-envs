"""Pin the base checkpoint and collect sha256 for plausible substitution targets.

The verifier rejects a submission whose weight file is bit-identical to any public
checkpoint other than the provided base. Note that HF only exposes sha256 for LFS
files, so small files such as config.json come back as None and are skipped.
"""

import json
from pathlib import Path

from huggingface_hub import HfApi

BASE = "DeepChem/ChemBERTa-77M-MLM"
REPOS = [
    BASE,
    "DeepChem/ChemBERTa-77M-MTR",
    "DeepChem/ChemBERTa-10M-MLM",
    "DeepChem/ChemBERTa-10M-MTR",
    "DeepChem/ChemBERTa-5M-MLM",
    "DeepChem/ChemBERTa-5M-MTR",
    "DeepChem/ChemBERTa-100M-MLM",
    "DeepChem/ChemBERTa-100M-MTR",
    "seyonec/ChemBERTa-zinc-base-v1",
    "ibm-research/MoLFormer-XL-both-10pct",
]


def main() -> None:
    api = HfApi()
    out = {}
    for repo in REPOS:
        try:
            info = api.model_info(repo, files_metadata=True)
        except Exception as exc:
            print(f"{repo:<40} SKIP ({type(exc).__name__})")
            continue
        files = {}
        for sib in info.siblings:
            entry = {"size": sib.size}
            if sib.lfs is not None:
                entry["sha256"] = sib.lfs.sha256
            files[sib.rfilename] = entry
        out[repo] = {"sha": info.sha, "files": files}
        weights = [f"{n}={m['sha256'][:12]}" for n, m in files.items()
                   if m.get("sha256") and n.endswith((".safetensors", ".bin"))]
        print(f"{repo:<40} rev={info.sha[:12]} {' '.join(weights) or '(no LFS weights)'}")

    dest = Path(__file__).parent / "results" / "public_hashes.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nbase revision: {out[BASE]['sha']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
