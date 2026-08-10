"""Measure reward variance across training seeds.

BBBP's scaffold test split is only 204 molecules, so seed-to-seed AUC noise could
plausibly rival the base-to-reference gap. If it does, a single small dataset
cannot carry the reward and the task needs an aggregate over several eval sets.

The requirement from the brief is that measured variance sits well below the
base-to-reference gap.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from moleculenet import fetch, finetune, morgan, multitask_auc, scaffold_split

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    df, labels, _ = fetch(args.dataset)
    smiles = df["smiles"].tolist()
    Y = df[labels].to_numpy(dtype=np.float64)
    tr, va, te, _ = scaffold_split(smiles)
    print(f"{args.dataset}: train={len(tr)} test={len(te)} tasks={len(labels)}")

    finals, bests = [], []
    for seed in args.seeds:
        curve, secs = finetune(smiles, Y, tr, va, te, args.epochs, args.threads,
                               seed=seed)
        finals.append(curve[-1])
        bests.append(max(curve))
        print(f"  seed {seed}: final={curve[-1]:.4f} best={max(curve):.4f} ({secs:.0f}s)",
              flush=True)

    out = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "final_auc": finals,
        "best_auc": bests,
        "final_mean": round(float(np.mean(finals)), 4),
        "final_std": round(float(np.std(finals)), 4),
        "final_range": round(float(max(finals) - min(finals)), 4),
        "best_mean": round(float(np.mean(bests)), 4),
        "best_std": round(float(np.std(bests)), 4),
    }
    print(f"\nfinal AUC: mean={out['final_mean']} std={out['final_std']} "
          f"range={out['final_range']}")
    print(f"best  AUC: mean={out['best_mean']} std={out['best_std']}")

    dest = HERE / "results" / f"variance_{args.dataset}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
