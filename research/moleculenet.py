"""Gate A for the molecular property track.

The question that killed Hydro, asked again: does the pretrained model beat the
trivial classical baseline? In cheminformatics that baseline is Morgan/ECFP
fingerprints plus a linear model or random forest, and it is genuinely strong on
MoleculeNet. If fingerprints match ChemBERTa on scaffold splits, this track dies
the same way Hydro did.

Scaffold splitting is the standard MoleculeNet protocol: group by Bemis-Murcko
scaffold, then fill train/valid/test from the largest scaffold groups down, so
test molecules have structural cores unseen in training.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data" / "moleculenet"
S3 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"

# name -> (filename, smiles column, label columns or None for "everything else")
DATASETS = {
    "bbbp": ("BBBP.csv", "smiles", ["p_np"]),
    "bace": ("bace.csv", "mol", ["Class"]),
    "clintox": ("clintox.csv.gz", "smiles", None),
    "sider": ("sider.csv.gz", "smiles", None),
    "tox21": ("tox21.csv.gz", "smiles", None),
}
DROP_COLS = {"smiles", "mol", "CID", "Model", "PUBCHEM_CID", "mol_id", "num", "name"}
BASE_MODEL = "DeepChem/ChemBERTa-77M-MLM"
SEED = 0


def fetch(name: str):
    fname, smi_col, labels = DATASETS[name]
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / fname
    if not dest.exists():
        urllib.request.urlretrieve(f"{S3}/{fname}", dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    df = pd.read_csv(dest)
    df = df.rename(columns={smi_col: "smiles"})
    if labels is None:
        labels = [c for c in df.columns if c not in DROP_COLS]
    return df[["smiles"] + labels], labels, digest


def scaffold_split(smiles, frac=(0.8, 0.1, 0.1)):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    groups = defaultdict(list)
    valid = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        valid.append(i)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups[scaf].append(i)

    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(valid)
    n_tr, n_va = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    tr, va, te = [], [], []
    for grp in ordered:
        if len(tr) + len(grp) <= n_tr:
            tr += grp
        elif len(tr) + len(va) + len(grp) <= n_va:
            va += grp
        else:
            te += grp
    return np.array(tr), np.array(va), np.array(te), len(groups)


def morgan(smiles, n_bits=2048, radius=2):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            X[i] = np.array(gen.GetFingerprint(mol), dtype=np.float32)
    return X


def multitask_auc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for t in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, t])
        if mask.sum() == 0 or len(np.unique(y_true[mask, t])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[mask, t], y_score[mask, t]))
    return float(np.mean(aucs)) if aucs else float("nan")


def fit_predict(model_fn, Xtr, Ytr, Xte):
    out = np.zeros((Xte.shape[0], Ytr.shape[1]))
    for t in range(Ytr.shape[1]):
        mask = ~np.isnan(Ytr[:, t])
        yt = Ytr[mask, t]
        if len(np.unique(yt)) < 2:
            continue
        out[:, t] = model_fn().fit(Xtr[mask], yt).predict_proba(Xte)[:, 1]
    return out


def chemberta_embed(smiles, threads: int, tag: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache = HERE / "cache" / f"cb_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    cache.parent.mkdir(exist_ok=True)

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL)
    model.eval()

    out, bs = [], 64
    with torch.no_grad():
        for i in range(0, len(smiles), bs):
            enc = tok(list(smiles[i : i + bs]), return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
    emb = np.concatenate(out).astype(np.float32)
    np.save(cache, emb)
    return emb


def finetune(smiles, Y, tr, va, te, epochs: int, threads: int, lr: float = 3e-5,
             random_init: bool = False, seed: int = SEED):
    import torch
    from transformers import (AutoConfig, AutoTokenizer,
                              RobertaForSequenceClassification)

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if random_init:
        # Identical architecture, no pretrained weights. If this matches the
        # pretrained run, the provided base model is not doing any work and the
        # task fails on validity the same way Hydro did.
        cfg = AutoConfig.from_pretrained(
            BASE_MODEL, num_labels=Y.shape[1],
            problem_type="multi_label_classification")
        model = RobertaForSequenceClassification(cfg)
    else:
        model = RobertaForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=Y.shape[1],
            problem_type="multi_label_classification")

    ytr = torch.tensor(np.nan_to_num(Y[tr], nan=0.0), dtype=torch.float32)
    wtr = torch.tensor(~np.isnan(Y[tr]), dtype=torch.float32)
    strs = [smiles[i] for i in tr]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bs = 32
    t0 = time.time()
    curve = []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(strs))
        for i in range(0, len(strs), bs):
            idx = perm[i : i + bs].tolist()
            enc = tok([strs[j] for j in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            logits = model(**enc).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, ytr[idx], weight=wtr[idx]
            )
            loss.backward()
            opt.step()
            opt.zero_grad()
        scores = predict(model, tok, [smiles[i] for i in te], threads)
        auc = multitask_auc(Y[te], scores)
        curve.append(round(auc, 4))
        print(f"    epoch {ep+1}/{epochs}: test AUC={auc:.4f} "
              f"({time.time()-t0:.0f}s, {(ep+1)*len(strs)/(time.time()-t0):.0f} mol/s)",
              flush=True)
    return curve, time.time() - t0


def predict(model, tok, smiles, threads: int, bs: int = 64):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), bs):
            enc = tok(smiles[i : i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            out.append(torch.sigmoid(model(**enc).logits).numpy())
    return np.concatenate(out)


def run(name: str, threads: int, epochs: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    df, labels, digest = fetch(name)
    smiles = df["smiles"].tolist()
    Y = df[labels].to_numpy(dtype=np.float64)
    tr, va, te, n_scaf = scaffold_split(smiles)

    print(f"\n{'='*72}\n{name}: {len(df):,} molecules, {len(labels)} task(s), "
          f"{n_scaf:,} scaffolds")
    print(f"  scaffold split: train={len(tr):,} valid={len(va):,} test={len(te):,}")
    res = {"dataset": name, "n": len(df), "n_tasks": len(labels),
           "n_scaffolds": n_scaf, "sha256": digest,
           "n_train": len(tr), "n_test": len(te)}

    t0 = time.time()
    F = morgan(smiles)
    fp_secs = time.time() - t0

    t0 = time.time()
    auc = multitask_auc(Y[te], fit_predict(
        lambda: LogisticRegression(max_iter=2000, C=1.0), F[tr], Y[tr], F[te]))
    res["morgan-logreg"] = round(auc, 4)
    print(f"  morgan+logreg   AUC={auc:.4f}  ({time.time()-t0+fp_secs:.1f}s)", flush=True)

    t0 = time.time()
    auc = multitask_auc(Y[te], fit_predict(
        lambda: RandomForestClassifier(n_estimators=300, n_jobs=threads, random_state=SEED),
        F[tr], Y[tr], F[te]))
    res["morgan-rf"] = round(auc, 4)
    print(f"  morgan+rf       AUC={auc:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    E = chemberta_embed(smiles, threads, name)
    emb_secs = time.time() - t0
    auc = multitask_auc(Y[te], fit_predict(
        lambda: LogisticRegression(max_iter=3000, C=1.0), E[tr], Y[tr], E[te]))
    res["chemberta-frozen"] = round(auc, 4)
    res["embed_seconds"] = round(emb_secs, 1)
    print(f"  chemberta-frozen AUC={auc:.4f}  (embed {emb_secs:.1f}s for {len(smiles):,} "
          f"= {len(smiles)/max(emb_secs,1e-9):.0f} mol/s)", flush=True)

    if epochs:
        for tag, rand in (("chemberta-finetune", False), ("randinit-finetune", True)):
            if rand and not run.include_random:
                continue
            print(f"  [{tag}]", flush=True)
            curve, secs = finetune(smiles, Y, tr, va, te, epochs, threads,
                                   random_init=rand)
            res[tag] = curve[-1]
            res[f"{tag}-best"] = max(curve)
            res[f"{tag}-curve"] = curve
            res[f"{tag}-seconds"] = round(secs, 1)
    return res


run.include_random = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["bbbp", "bace", "clintox"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--random-init-control", action="store_true",
                    help="also fine-tune an untrained copy of the same architecture")
    ap.add_argument("--out", default="results/moleculenet.json")
    args = ap.parse_args()

    run.include_random = args.random_init_control
    print("downloading MoleculeNet ...")
    results = [run(d, args.threads, args.epochs) for d in args.datasets]

    dest = HERE / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2))

    print(f"\n{'dataset':<10} {'morgan-lr':>10} {'morgan-rf':>10} {'cb-frozen':>10} "
          f"{'cb-ft':>8} {'rand-ft':>8} {'pretrain gain':>14}")
    for r in results:
        cb = r.get("chemberta-finetune-best", float("nan"))
        rd = r.get("randinit-finetune-best", float("nan"))
        print(f"{r['dataset']:<10} {r['morgan-logreg']:>10.4f} {r['morgan-rf']:>10.4f} "
              f"{r['chemberta-frozen']:>10.4f} {cb:>8.4f} {rd:>8.4f} {cb - rd:>+14.4f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
