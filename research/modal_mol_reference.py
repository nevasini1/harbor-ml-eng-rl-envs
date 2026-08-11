"""Re-measure the mol task's `reference` anchor, because an agent cleared it.

Why
---
`codex` scored reward 1.0 on its first attempt, at uncapped recovery 1.280 on
tox21 and 1.472 on bbbp. Its tox21 AUC of 0.7209 also exceeds the best of the 25
seeded runs (0.7111) that calibrated the anchor. Both eval sets clipped, so the
reward stopped discriminating: every submission from 0.7019 upward on tox21 scores
exactly 1.0, and there is no gradient left above the bar.

A reference is defined as "a tuned, deliberately ordinary adaptation" -- a bar a
strong agent can exceed, but not on a first attempt by half a band. The current one
no longer satisfies its own definition, so it is re-measured rather than edited:
editing the constant would produce a number no script in this repo emits, which is
the exact failure the protein task is being repaired for.

The arms
--------
Three candidate references, chosen from what the agent's own log
(`jobs/mol-codex-modal/.../agent/train_log.txt`) shows mattered:

  current    the shipped recipe, transcribed from the two scripts that measured
             the anchors: body lr 3e-5, head 1e-3, batch 32, OneCycle, 20 epochs,
             best epoch on a RANDOM 20% validation slice, loss normalised by the
             observed-label count. This arm is a CONTROL -- if it does not land on
             tox21 0.7019 / bbbp 0.9121, nothing else in the table is comparable
             to the shipped scale and the run is void.

             The first run of this file was void, and the control is why we know.
             It had batch 16, a 10% slice, and `weight=`-reduced loss (divide by
             all labels rather than observed ones). It returned tox21 0.6830 and
             bbbp 0.9153 -- and the split between them is the diagnosis: bbbp is
             100% observed so the loss bug cannot touch it, tox21 is 83.8% observed
             so every gradient there was scaled by 0.838. Reading that table would
             have shown "the shipped tox21 reference is 0.019 optimistic", which is
             a claim about my transcription, not about the task.

  grouped    identical, except validation is a held-out set of whole Bemis-Murcko
             scaffold GROUPS. This is the single change the agent's log points at
             hardest: it built "group-held-out validation folds to approximate the
             stated chemical-space shift", while the shipped oracle selects on a
             random slice whose distribution does not match the graded holdout.
             Selecting on the wrong distribution is a real defect, not a tuning
             preference.

  stronger   grouped validation plus the agent's hyperparameters: body 5e-5,
             head 4e-4, batch 32, cosine schedule with 10% warmup, and fourth-root
             inverse-frequency positive weights on tox21 (its log measured
             none 0.6883 / fourth-root 0.6898 / sqrt 0.6884).

Deliberately NOT included: randomized-SMILES augmentation and the BBBP->Tox21
encoder transfer. Both helped the agent, and both are exactly the kind of insight
the task should still be rewarding an agent for finding. A reference that already
contains every trick leaves nothing to measure.

    modal run research/modal_mol_reference.py

Writes results/mol_reference_candidates.json. Choosing among the arms, and
re-deriving the anchors, is a separate step -- this file measures.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent
REMOTE = "/work"
TASK = "tasks/mol-property-adapt"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers==4.49.0", "safetensors==0.4.5",
                 "scikit-learn==1.5.2", "pandas==2.2.3", "numpy==1.26.4",
                 "scipy==1.14.1", "huggingface_hub==0.28.1", "rdkit==2024.9.4")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(f"{ROOT}/{TASK}/environment/data", f"{REMOTE}/train")
    .add_local_dir(f"{ROOT}/{TASK}/tests/grader/private", f"{REMOTE}/private")
)
cache = modal.Volume.from_name("posttrain-hf-cache", create_if_missing=True)
app = modal.App("mol-reference")

BASE = "DeepChem/ChemBERTa-77M-MLM"
BASE_REV = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"

# `current` must reproduce the shipped anchor, so every value in it is copied from
# the scripts that measured it -- modal_legal_anchors.py:arm_legal_finetune (tox21)
# and modal_bbbp_split.py (bbbp), which agree on all of them. The other two arms
# change one thing at a time from there.
ARMS = {
    "current":  dict(body_lr=3e-5, head_lr=1e-3, bs=32, epochs=20, val_frac=0.2,
                     sched="onecycle", grouped_val=False, weighting=None),
    "grouped":  dict(body_lr=3e-5, head_lr=1e-3, bs=32, epochs=20, val_frac=0.2,
                     sched="onecycle", grouped_val=True,  weighting=None),
    "stronger": dict(body_lr=5e-5, head_lr=4e-4, bs=32, epochs=20, val_frac=0.2,
                     sched="cosine", grouped_val=True,  weighting="fourth_root"),
}


@app.function(image=image, gpu="A10G", timeout=60 * 60,
              volumes={"/cache": cache}, max_containers=12)
def run(arm: str, eval_set: str, seed: int) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = ARMS[arm]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tr = pd.read_csv(f"{REMOTE}/train/{eval_set}_train.csv")
    te = pd.read_csv(f"{REMOTE}/private/{eval_set}_test.csv")
    labels = [c for c in tr.columns if c != "smiles"]
    smiles = tr["smiles"].astype(str).tolist()
    Y = tr[labels].to_numpy(dtype=np.float64)

    # ---- validation split: random slice, or whole scaffold groups held out.
    n_val = max(1, int(cfg["val_frac"] * len(smiles)))
    if cfg["grouped_val"]:
        from collections import defaultdict

        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")
        groups = defaultdict(list)
        for i, smi in enumerate(smiles):
            m = Chem.MolFromSmiles(smi)
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) \
                if m is not None else f"__bad{i}"
            groups[key].append(i)
        order = rng.permutation(len(groups))
        keys = list(groups)
        val_idx: list[int] = []
        for gi in order:                      # whole groups, until the slice is full
            if len(val_idx) >= n_val:
                break
            val_idx += groups[keys[gi]]
        # NOT truncated to exactly n_val: cutting the last group in half would put
        # the same scaffold on both sides, which is the leak this arm exists to close.
        val_idx = np.array(val_idx)
    else:
        val_idx = rng.permutation(len(smiles))[:n_val]
    fit_idx = np.array([i for i in range(len(smiles)) if i not in set(val_idx.tolist())])

    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=BASE_REV, num_labels=len(labels),
        problem_type="multi_label_classification").to(dev)

    # ---- fourth-root inverse-frequency positive weights (tox21 is imbalanced)
    pos_w = None
    if cfg["weighting"] == "fourth_root":
        w = []
        for t in range(len(labels)):
            col = Y[fit_idx, t]
            col = col[~np.isnan(col)]
            pos, neg = max((col == 1).sum(), 1), max((col == 0).sum(), 1)
            w.append((neg / pos) ** 0.25)
        pos_w = torch.tensor(w, dtype=torch.float32, device=dev)

    head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": cfg["body_lr"]},
                             {"params": head, "lr": cfg["head_lr"]}], weight_decay=0.01)
    steps = cfg["epochs"] * max(1, -(-len(fit_idx) // cfg["bs"]))   # ceil, as the anchor scripts do
    if cfg["sched"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[cfg["body_lr"], cfg["head_lr"]], total_steps=steps,
            pct_start=0.1, anneal_strategy="linear")
    else:
        warm = max(1, int(0.1 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warm if s < warm
            else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))

    def predict(idx_smiles):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx_smiles), 64):
                enc = tok(idx_smiles[i:i + 64], return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(dev)
                out.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
        model.train()
        return np.concatenate(out)

    def mean_auc(y, p):
        a = []
        for t in range(y.shape[1]):
            m = ~np.isnan(y[:, t])
            if m.sum() and len(np.unique(y[m, t])) == 2:
                a.append(roc_auc_score(y[m, t], p[m, t]))
        return float(np.mean(a)) if a else float("nan")

    yfit = torch.tensor(np.nan_to_num(Y[fit_idx], nan=0.0), dtype=torch.float32, device=dev)
    wfit = torch.tensor(~np.isnan(Y[fit_idx]), dtype=torch.float32, device=dev)
    fit_smiles = [smiles[i] for i in fit_idx]
    val_smiles = [smiles[i] for i in val_idx]

    best, best_state, best_ep, step = -1.0, None, -1, 0
    for ep in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(len(fit_idx))
        for i in range(0, len(fit_idx), cfg["bs"]):
            j = perm[i:i + cfg["bs"]]
            enc = tok([fit_smiles[k] for k in j], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(dev)
            logits = model(**enc).logits
            # Normalised by the number of OBSERVED labels, not by all of them.
            # tox21 is 83.8% observed, so `weight=`'s divide-by-numel would shrink
            # every gradient by that factor on tox21 and not at all on bbbp.
            per = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yfit[j], reduction="none", pos_weight=pos_w)
            loss = (per * wfit[j]).sum() / wfit[j].sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step < steps:
                sched.step()
        v = mean_auc(Y[val_idx], predict(val_smiles))
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    auc = mean_auc(te[labels].to_numpy(dtype=np.float64),
                   predict(te["smiles"].astype(str).tolist()))
    print(f"{arm}/{eval_set}/seed{seed}: test_auc={auc:.4f} "
          f"(val {best:.4f} @ epoch {best_ep + 1})", flush=True)
    return {"arm": arm, "eval_set": eval_set, "seed": seed, "auc": auc,
            "val": best, "best_epoch": best_ep + 1}


@app.local_entrypoint()
def main(seeds: int = 5, arms: str = "current,grouped,stronger",
         eval_sets: str = "tox21,bbbp"):
    import statistics as st

    grid = [(a, e, s) for a in arms.split(",") for e in eval_sets.split(",")
            for s in range(seeds)]
    print(f"{len(grid)} containers")
    rows = [r for r in run.starmap(grid) if r]

    out = {"arms": {}, "note": "candidate references for the mol task; see the "
                               "module docstring for why each arm exists"}
    for a in arms.split(","):
        out["arms"][a] = {}
        for e in eval_sets.split(","):
            v = [r["auc"] for r in rows if r["arm"] == a and r["eval_set"] == e]
            if v:
                out["arms"][a][e] = {"mean": round(st.fmean(v), 4),
                                     "std": round(st.stdev(v), 4) if len(v) > 1 else 0.0,
                                     "seeds": [round(x, 4) for x in v]}
    dest = HERE / "results" / "mol_reference_candidates.json"   # written once, below,
                                                                # after the control verdict
    print(f"\n{'arm':<12}" + "".join(f"{e:>22}" for e in eval_sets.split(",")))
    for a, d in out["arms"].items():
        print(f"{a:<12}" + "".join(
            f"{d[e]['mean']:>15.4f}±{d[e]['std']:.3f}" if e in d else f"{'-':>22}"
            for e in eval_sets.split(",")))
    # ---- the control gate. Stated before the numbers are read, not after.
    shipped = {"tox21": 0.7019, "bbbp": 0.9121}
    print(f"\nshipped reference: tox21 0.7019, bbbp 0.9121")
    print(f"codex agent:       tox21 0.7209, bbbp 0.9188")

    ctl = out["arms"].get("current", {})
    verdict = []
    for e, want in shipped.items():
        if e not in ctl:
            continue
        got, sd = ctl[e]["mean"], ctl[e]["std"]
        sem = sd / max(1, len(ctl[e]["seeds"])) ** 0.5
        z = abs(got - want) / sem if sem else float("inf")
        verdict.append(z <= 2.0)
        print(f"  control {e}: {got:.4f} vs shipped {want:.4f} -> {z:.1f} sem "
              f"({'reproduces' if z <= 2.0 else 'DOES NOT REPRODUCE'})")
    out["control_reproduces_shipped_anchor"] = bool(verdict) and all(verdict)
    if not out["control_reproduces_shipped_anchor"]:
        print("\n  VOID: the `current` arm is not the shipped recipe. The other arms\n"
              "  are still comparable to each other, but not to the shipped anchors.")
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
