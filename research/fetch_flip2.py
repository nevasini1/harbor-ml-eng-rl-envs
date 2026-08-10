"""Download and checksum-verify the FLIP2 subsets used by the spike.

Pinned to Zenodo record 18433203 (DOI 10.5281/zenodo.18433203, CC-BY 4.0).
Only the files listed in WANTED are fetched; md5s are taken from the record
manifest and re-verified locally so the download can be pinned in a Dockerfile.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORD = "18433203"
API = f"https://zenodo.org/api/records/{RECORD}"
DATA = Path(__file__).parent / "data" / "flip2"

WANTED = [
    "hydro/README.md",
    "hydro/random_split.csv.gz",
    "hydro/low_to_high.csv.gz",
    "hydro/three_to_many.csv.gz",
    "hydro/to_P06241.csv.gz",
    "hydro/to_P01053.csv.gz",
    "hydro/to_P0A9X9.csv.gz",
    "gb1/README.md",
    "gb1/sampled.csv.gz",
    "gb1/one_vs_rest.csv.gz",
    "gb1/two_vs_rest.csv.gz",
    "gb1/three_vs_rest.csv.gz",
    "gb1/low_vs_high.csv.gz",
    "meltome/README.md",
    "meltome/mixed_split.csv.gz",
    "pdz3/README.md",
    "pdz3/rand_split.csv.gz",
    "pdz3/single_to_double.csv.gz",
    "rhomax/README.md",
    "rhomax/by_wild_type.csv.gz",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    with urllib.request.urlopen(API, timeout=60) as resp:
        record = json.load(resp)
    manifest = {f["key"]: f for f in record["files"]}

    missing = [k for k in WANTED if k not in manifest]
    if missing:
        print(f"NOT IN RECORD: {missing}", file=sys.stderr)
        return 1

    pins = {}
    for key in WANTED:
        entry = manifest[key]
        want = entry["checksum"].split(":", 1)[1]
        dest = DATA / key
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and md5(dest) == want:
            print(f"cached  {key}")
        else:
            urllib.request.urlretrieve(entry["links"]["self"], dest)
            got = md5(dest)
            if got != want:
                print(f"CHECKSUM MISMATCH {key}: {got} != {want}", file=sys.stderr)
                return 1
            print(f"ok      {key}  {entry['size']:>9,} B")
        pins[key] = {"md5": want, "size": entry["size"], "url": entry["links"]["self"]}

    pinfile = DATA / "PINS.json"
    pinfile.write_text(
        json.dumps(
            {"record": RECORD, "doi": record["doi"], "license": "cc-by-4.0", "files": pins},
            indent=2,
        )
    )
    print(f"\n{len(pins)} files verified, pins written to {pinfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
