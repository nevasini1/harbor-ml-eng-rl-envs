"""Rebuild the bbbp private split, selecting the anchor on measured band.

Why
---
bbbp's shipped private split is the rare-scaffold tail, which region_split.py's
own docstring lists as REJECTED ("Rare-scaffold tail, privately selected - AUC
0.967"). The manifest confirms it: tail_frac 0.35, n_test_groups 407. Measured
consequence: frozen logreg 0.9671, frozen head 0.9668, fine-tune 0.9677 -- a
+0.0006 spread at 0.34 sigma, with within-method noise larger than the
between-method gap. Nothing is scoreable on it.

The dataset is not the problem. On the public scaffold split SPIKE_RESULTS.md
rates bbbp the *primary* eval set, with the cleanest ladder in the whole spike:
random-init 0.667 < Morgan+RF 0.705 < frozen probe 0.726 < fine-tune 0.744, and
+0.077 of pretraining gain. So bbbp is worth rebuilding rather than discarding;
only the split needs replacing, with the same Tanimoto anchor-region holdout that
tox21 uses.

Selection rule -- v2
--------------------
write_region_split.py records the old rule: "lowest frozen-probe AUC among fully
scoreable candidates", i.e. pick the hardest region. This session showed that is
the wrong objective. Difficulty is not headroom: the protein task is very hard
and completely inverted, and a region can be hard for the probe while being hard
for the fine-tune too, leaving nothing in between.

v1 ranked candidates by raw band and picked anchor 938: band +0.0356 but only
3.77 sigma, because its test set was 89.2% positive and single-task AUC over ~44
negatives is unstable. Raw band was itself a proxy. Three corrections:

  * rank on band / pooled noise, not band. Separation is the quantity that
    decides whether a reward is measuring the submission or the seed.
  * require MIN_MINORITY_IN_TEST molecules of the scarcer class, which bounds
    AUC variance at the source rather than hoping the search avoids it.
  * fine-tune every surviving candidate. v1 pre-filtered to the top 5 by lowest
    frozen probe -- the same difficulty-as-headroom heuristic this file argues
    against, left in as a cost-saving proxy.

Both arms are legal submissions under the verifier's contract
(RobertaForSequenceClassification reading the CLS token), because an anchor
measured with any other pooling is unreachable by an agent -- the mistake that
made base 0.6430 / reference 0.7324 unusable for tox21.

Stages
------
  1  CLS-embed every molecule once. Embeddings depend on the molecule, not the
     partition, so each candidate split is then just an index operation.
  2  Screen N candidate anchors on the two frozen arms (cheap, cached embeddings).
     Drop any whose holdout is not scoreable or is too class-imbalanced.
  3  Fine-tune every survivor, SCREEN_SEEDS each, both arms -> band and sigma.
  4  Re-measure the winner at FINAL_SEEDS to produce shippable anchors.

Molecules come from the union of the two existing bbbp CSVs rather than a fresh
MoleculeNet fetch, so the source cannot drift underneath the rebuild.

Run:  modal run tasks/sciml-protein-regression/scripts/modal_bbbp_split.py
"""

from __future__ import annotations

import modal

BASE = "DeepChem/ChemBERTa-77M-MLM"
REVISION = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
TEST_FRAC = 0.2
N_CANDIDATES = 40
SCREEN_SEEDS = 3
FINAL_SEEDS = 5

# AUC is estimated from positive-negative pairs, so its variance is governed by
# whichever class is scarcer. The v1 winner (anchor 938) had a test set 89.2%
# positive -- about 44 negatives out of 407 -- and its fine-tune arm came out at
# 24.7% of band against tox21's 8.1%. Requiring 100 minority-class molecules
# bounds that directly instead of hoping a band-maximising search avoids it.
MIN_MINORITY_IN_TEST = 100
EPOCHS = 20
BS = 32
BODY_LR = 3e-5
HEAD_LR = 1e-3

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0", "transformers==4.49.0", "scikit-learn==1.5.2",
        "pandas==2.2.3", "numpy==1.26.4", "scipy==1.14.1", "rdkit==2024.3.5",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_file("research/split/agent/bbbp_train.csv", "/data/bbbp_a.csv")
    .add_local_file("research/split/private/bbbp_test.csv", "/data/bbbp_b.csv")
    .add_local_file("research/PRIVATE_SEED", "/data/PRIVATE_SEED")
)

cache = modal.Volume.from_name("esm-probe-cache", create_if_missing=True)
app = modal.App("bbbp-split-rebuild")


def _all_molecules():
    """Union of the shipped agent/private halves = the full deduped bbbp set."""
    import pandas as pd

    a = pd.read_csv("/data/bbbp_a.csv")
    b = pd.read_csv("/data/bbbp_b.csv")
    df = pd.concat([a, b], ignore_index=True).drop_duplicates("smiles")
    df = df.reset_index(drop=True)
    labels = [c for c in df.columns if c != "smiles"]
    return df, labels


def _morgan_bits(smiles):
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        X[i] = np.array(gen.GetFingerprint(m), dtype=np.float32)
    return (X > 0).astype(np.float32)


def _tanimoto_to(F, anchor):
    import numpy as np

    inter = F @ anchor
    union = F.sum(1) + anchor.sum() - inter
    return inter / np.maximum(union, 1e-9)


def _partition(F, anchor_i, n_test):
    """Hold out the whole region of chemical space nearest the anchor."""
    import numpy as np

    sim = _tanimoto_to(F, F[anchor_i])
    order = np.argsort(-sim)
    return np.sort(order[:n_test]), np.sort(order[n_test:]), sim


def _mean_auc(y_true, y_score) -> float:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    aucs = []
    for j in range(y_true.shape[1]):
        obs = ~np.isnan(y_true[:, j])
        yt = y_true[obs, j]
        if obs.sum() < 10 or yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt, y_score[obs, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _scoreable(Y, idx) -> int:
    import numpy as np

    n = 0
    for t in range(Y.shape[1]):
        m = ~np.isnan(Y[idx, t])
        if m.sum() >= 20 and len(np.unique(Y[idx, t][m])) == 2:
            n += 1
    return n


def _frozen_head_auc(Xtr, Ytr, Mtr, Xte, Yte, seed):
    """Legal trivial ceiling: RobertaClassificationHead on the CLS embedding."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Ytr))
    fit_i, val_i = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)) :]

    torch.manual_seed(seed)
    hid = Xtr.shape[1]

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(hid, hid)
            self.dropout = torch.nn.Dropout(0.1)
            self.out_proj = torch.nn.Linear(hid, Ytr.shape[1])

        def forward(self, x):
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    X = torch.tensor(Xtr, dtype=torch.float32, device="cuda")
    Xt = torch.tensor(Xte, dtype=torch.float32, device="cuda")
    Y = torch.tensor(np.nan_to_num(Ytr), dtype=torch.float32, device="cuda")
    M = torch.tensor(Mtr, dtype=torch.float32, device="cuda")

    head = Head().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=1e-2)
    best_val, best_te, patience = -1.0, None, 0
    for _ in range(300):
        head.train()
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            head(X[fit_i]), Y[fit_i], reduction="none") * M[fit_i]).sum() / M[fit_i].sum()
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            va = _mean_auc(Ytr[val_i], head(X[val_i]).cpu().numpy())
        if va > best_val + 1e-5:
            best_val, patience = va, 0
            with torch.no_grad():
                best_te = head(Xt).cpu().numpy()
        else:
            patience += 1
            if patience >= 20:
                break
    return _mean_auc(Yte, best_te)


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/cache": cache})
def embed_and_screen() -> dict:
    """Stages 1-2: embed once, then score every candidate anchor on frozen arms."""
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModel, AutoTokenizer

    df, labels = _all_molecules()
    smiles = df["smiles"].tolist()
    Y = df[labels].to_numpy(np.float64)
    n_test = int(TEST_FRAC * len(df))
    print(f"bbbp: {len(df)} molecules, {len(labels)} task(s), n_test={n_test}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    enc_model = AutoModel.from_pretrained(BASE, revision=REVISION).cuda().eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(smiles), 128):
            enc = tok(smiles[i : i + 128], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            embs.append(enc_model(**enc).last_hidden_state[:, 0, :].float().cpu())
    E = torch.cat(embs).numpy()

    F = _morgan_bits(smiles)
    seed = int(open("/data/PRIVATE_SEED").read().strip())
    rng = np.random.default_rng(seed)
    candidates = rng.choice(len(df), size=N_CANDIDATES, replace=False)

    rows = []
    for a in candidates:
        te_i, tr_i, sim = _partition(F, int(a), n_test)
        n_ok = _scoreable(Y, te_i)
        if n_ok < 1:
            print(f"anchor {a}: holdout not scoreable, dropped", flush=True)
            continue
        yte_obs = Y[te_i][~np.isnan(Y[te_i])]
        n_pos = int((yte_obs == 1).sum())
        n_minority = min(n_pos, len(yte_obs) - n_pos)
        if n_minority < MIN_MINORITY_IN_TEST:
            print(f"anchor {a}: only {n_minority} minority-class test molecules "
                  f"(need {MIN_MINORITY_IN_TEST}), dropped", flush=True)
            continue

        scores = np.zeros((len(te_i), Y.shape[1]))
        for j in range(Y.shape[1]):
            obs = ~np.isnan(Y[tr_i, j])
            yj = Y[tr_i, j][obs]
            if yj.min() == yj.max():
                continue
            clf = LogisticRegression(max_iter=3000, random_state=0)
            clf.fit(E[tr_i][obs], yj)
            scores[:, j] = clf.predict_proba(E[te_i])[:, 1]
        lg = _mean_auc(Y[te_i], scores)

        hd = _frozen_head_auc(E[tr_i], Y[tr_i], ~np.isnan(Y[tr_i]),
                              E[te_i], Y[te_i], seed=0)
        rows.append({
            "anchor": int(a), "n_scoreable_tasks": n_ok,
            "frozen_logreg": round(float(lg), 4),
            "frozen_head": round(float(hd), 4),
            "best_frozen": round(float(max(lg, hd)), 4),
            "n_minority_in_test": n_minority,
            "n_train": int(len(tr_i)), "n_test": int(len(te_i)),
            "test_pos_rate": round(float(np.nanmean(Y[te_i])), 4),
            "train_pos_rate": round(float(np.nanmean(Y[tr_i])), 4),
            "mean_sim_in_test": round(float(sim[te_i].mean()), 4),
        })
        print(f"anchor {a}: logreg {lg:.4f} head {hd:.4f} "
              f"test_pos {rows[-1]['test_pos_rate']:.3f}", flush=True)

    np.savez("/cache/bbbp_cls_emb.npz", E=E, F=F)
    cache.commit()
    rows.sort(key=lambda r: r["best_frozen"])
    return {"n_molecules": len(df), "n_test": n_test, "labels": labels,
            "candidates": rows}


@app.function(gpu="A10G", image=image, timeout=5400, volumes={"/cache": cache})
def finetune_anchor(anchor: int, seed: int) -> dict:
    """Legal reference arm for one candidate partition."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df, labels = _all_molecules()
    smiles = df["smiles"].tolist()
    Y = df[labels].to_numpy(np.float64)
    n_test = int(TEST_FRAC * len(df))

    z = np.load("/cache/bbbp_cls_emb.npz")
    te_i, tr_i, _ = _partition(z["F"], anchor, n_test)
    tr_s = [smiles[i] for i in tr_i]
    te_s = [smiles[i] for i in te_i]
    ytr, yte = Y[tr_i], Y[te_i]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    fit_i, val_i = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)) :]

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=ytr.shape[1],
        problem_type="multi_label_classification").cuda()
    head_p = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    body_p = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW(
        [{"params": body_p, "lr": BODY_LR}, {"params": head_p, "lr": HEAD_LR}],
        weight_decay=0.01)
    total_steps = EPOCHS * max(1, -(-len(fit_i) // BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[BODY_LR, HEAD_LR], total_steps=total_steps, pct_start=0.1,
        anneal_strategy="linear")

    Yt = torch.tensor(np.nan_to_num(ytr), dtype=torch.float32, device="cuda")
    Mt = torch.tensor(~np.isnan(ytr), dtype=torch.float32, device="cuda")

    def forward(sm, train):
        enc = tok(sm, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to("cuda")
        with torch.enable_grad() if train else torch.no_grad():
            return model(**enc).logits

    def evaluate(sm, y):
        model.eval()
        outs = [forward(sm[i : i + 128], False).float().cpu()
                for i in range(0, len(sm), 128)]
        return _mean_auc(y, torch.cat(outs).numpy())

    step, best_val, best_te = 0, -1.0, -1.0
    for _ in range(EPOCHS):
        model.train()
        order = rng.permutation(fit_i)
        for i in range(0, len(order), BS):
            b = order[i : i + BS]
            opt.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                forward([tr_s[k] for k in b], True), Yt[b],
                reduction="none") * Mt[b]).sum() / Mt[b].sum()
            loss.backward()
            opt.step()
            step += 1
            if step < total_steps:
                sched.step()
        va = evaluate([tr_s[k] for k in val_i], ytr[val_i])
        if va > best_val:
            best_val = va
            best_te = evaluate(te_s, yte)
    print(f"anchor {anchor} seed {seed} finetune -> {best_te:.4f}", flush=True)
    return {"anchor": anchor, "seed": seed, "finetune_auc": best_te,
            "val_auc": best_val}


@app.function(gpu="A10G", image=image, timeout=1800, volumes={"/cache": cache})
def frozen_head_anchor(anchor: int, seed: int) -> dict:
    import numpy as np

    df, labels = _all_molecules()
    Y = df[labels].to_numpy(np.float64)
    n_test = int(TEST_FRAC * len(df))
    z = np.load("/cache/bbbp_cls_emb.npz")
    te_i, tr_i, _ = _partition(z["F"], anchor, n_test)
    auc = _frozen_head_auc(z["E"][tr_i], Y[tr_i], ~np.isnan(Y[tr_i]),
                           z["E"][te_i], Y[te_i], seed=seed)
    print(f"anchor {anchor} seed {seed} frozen_head -> {auc:.4f}", flush=True)
    return {"anchor": anchor, "seed": seed, "frozen_head_auc": auc}


@app.local_entrypoint()
def main(confirm_anchor: int = -1):
    """With --confirm-anchor N, skip the screen and measure one anchor at
    FINAL_SEEDS. The 3-seed screen ranks on a std estimated from n=3, which is
    itself unstable: anchor 1026 screened at 7.09 sigma and re-measured at 5.34.
    Runners-up therefore deserve a full measurement before a split is written."""
    import json
    import statistics as st
    from pathlib import Path

    def stat(v):
        return {"n": len(v), "mean": round(st.mean(v), 4),
                "std": round(st.stdev(v), 4) if len(v) > 1 else 0.0,
                "min": round(min(v), 4), "max": round(max(v), 4)}

    if confirm_anchor >= 0:
        ft = stat([r["finetune_auc"] for r in finetune_anchor.starmap(
            [(confirm_anchor, s) for s in range(FINAL_SEEDS)])])
        hd = stat([r["frozen_head_auc"] for r in frozen_head_anchor.starmap(
            [(confirm_anchor, s) for s in range(FINAL_SEEDS)])])
        band = ft["mean"] - hd["mean"]
        pooled = (hd["std"] ** 2 + ft["std"] ** 2) ** 0.5
        sigma = round(band / pooled, 2) if pooled else None
        res = {"anchor": confirm_anchor, "n_seeds": FINAL_SEEDS,
               "legal_frozen_head": hd, "legal_finetune": ft,
               "base_auc": hd["mean"], "reference_auc": ft["mean"],
               "band": round(band, 4), "band_sigma": sigma}
        dest = Path(__file__).resolve().parent / f"bbbp_confirm_{confirm_anchor}.json"
        dest.write_text(json.dumps(res, indent=2) + "\n")
        print("=" * 70)
        print(f"anchor {confirm_anchor}: base {hd['mean']:.4f} +/- {hd['std']:.4f}  "
              f"reference {ft['mean']:.4f} +/- {ft['std']:.4f}")
        print(f"band {band:+.4f}  ({sigma} sigma)")
        print(f"\nanchor 1026 for contrast: band +0.0187, 5.34 sigma")
        print(f"wrote {dest}")
        return

    screen = embed_and_screen.remote()
    cands = screen["candidates"]
    if not cands:
        print("no candidate anchor is both scoreable and balanced enough; "
              "bbbp cannot be re-split this way")
        return

    anchors = [c["anchor"] for c in cands]
    print(f"\n{len(anchors)} of {N_CANDIDATES} candidates survived the balance "
          f"filter (>= {MIN_MINORITY_IN_TEST} minority-class test molecules)")
    print(f"fine-tuning all of them at {SCREEN_SEEDS} seeds: {anchors}\n")

    jobs = [(a, s) for a in anchors for s in range(SCREEN_SEEDS)]
    ft_rows = list(finetune_anchor.starmap(jobs))
    hd_rows = list(frozen_head_anchor.starmap(jobs))

    ranked = []
    for a in anchors:
        ft = stat([r["finetune_auc"] for r in ft_rows if r["anchor"] == a])
        hd = stat([r["frozen_head_auc"] for r in hd_rows if r["anchor"] == a])
        band = ft["mean"] - hd["mean"]
        pooled = (hd["std"] ** 2 + ft["std"] ** 2) ** 0.5
        c = next(x for x in cands if x["anchor"] == a)
        ranked.append({
            "anchor": a, "frozen_head": hd, "finetune": ft,
            "band": round(band, 4),
            # Separation, not band, is the objective: it is what decides whether
            # the reward is measuring the submission or the seed.
            "band_sigma": round(band / pooled, 2) if pooled else None,
            "n_minority_in_test": c["n_minority_in_test"],
            "test_pos_rate": c["test_pos_rate"],
        })
    ranked.sort(key=lambda r: (r["band_sigma"] or -1), reverse=True)

    print(f"{'anchor':<9}{'minority':<10}{'frozen':<9}{'finetune':<10}"
          f"{'band':<9}{'sigma':<7}")
    for r in ranked:
        print(f"{r['anchor']:<9}{r['n_minority_in_test']:<10}"
              f"{r['frozen_head']['mean']:<9.4f}{r['finetune']['mean']:<10.4f}"
              f"{r['band']:<+9.4f}{r['band_sigma'] or float('nan'):<7.2f}")

    winner = ranked[0]["anchor"]
    print(f"\nwinner: anchor {winner} at {ranked[0]['band_sigma']} sigma; "
          f"confirming at {FINAL_SEEDS} seeds...\n")

    fin_ft = list(finetune_anchor.starmap([(winner, s) for s in range(FINAL_SEEDS)]))
    fin_hd = list(frozen_head_anchor.starmap([(winner, s) for s in range(FINAL_SEEDS)]))
    ft = stat([r["finetune_auc"] for r in fin_ft])
    hd = stat([r["frozen_head_auc"] for r in fin_hd])
    band = ft["mean"] - hd["mean"]
    pooled = (hd["std"] ** 2 + ft["std"] ** 2) ** 0.5
    sigma = round(band / pooled, 2) if pooled else None

    out = {
        "version": 2,
        "selection_rule": "largest band / pooled-noise (separation) among candidates "
                          "with >= %d minority-class test molecules; every survivor "
                          "fine-tuned, no frozen-probe pre-filter" % MIN_MINORITY_IN_TEST,
        "contract": "RobertaForSequenceClassification + .logits (CLS head)",
        "n_molecules": screen["n_molecules"], "n_test": screen["n_test"],
        "n_candidates_screened": N_CANDIDATES,
        "n_survived_balance_filter": len(anchors),
        "winning_anchor": winner,
        "screen": cands,
        "ranked": ranked,
        "final": {"legal_frozen_head": hd, "legal_finetune": ft},
        "proposed_anchors": {
            "base_auc": hd["mean"], "reference_auc": ft["mean"],
            "band": round(band, 4), "band_sigma": sigma,
        },
        "noise_as_pct_of_band": {
            "frozen_head": round(hd["std"] / band * 100, 1) if band > 0 else None,
            "finetune": round(ft["std"] / band * 100, 1) if band > 0 else None,
        },
        "for_contrast": {
            "tox21_on_contract": {"band": 0.0678, "band_sigma": 6.48,
                                  "finetune_noise_pct": 8.1},
            "bbbp_v1_band_selected": {"anchor": 938, "band": 0.0356,
                                      "band_sigma": 3.77, "finetune_noise_pct": 24.7},
            "bbbp_shipped_rare_scaffold_tail": {"band": 0.0006, "band_sigma": 0.34},
        },
        "raw": {"screen_finetune": ft_rows, "screen_frozen_head": hd_rows,
                "final_finetune": fin_ft, "final_frozen_head": fin_hd},
    }
    dest = Path(__file__).resolve().parent / "bbbp_split_v2.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print("=" * 70)
    print(f"anchor {winner}: base {hd['mean']:.4f} +/- {hd['std']:.4f}  "
          f"reference {ft['mean']:.4f} +/- {ft['std']:.4f}")
    print(f"band {band:+.4f}  ({sigma} sigma)")
    print(f"noise: frozen_head {out['noise_as_pct_of_band']['frozen_head']}% of band, "
          f"finetune {out['noise_as_pct_of_band']['finetune']}%")
    print(f"\ntox21 for contrast: band +0.0678, 6.48 sigma, finetune noise 8.1%")
    print(f"bbbp v1 (band-selected): band +0.0356, 3.77 sigma, finetune noise 24.7%")
    print(f"\nwrote {dest}")
    if band <= 0:
        print("\n=> inverted: drop bbbp.")
    elif (sigma or 0) < 3:
        print("\n=> under 3 sigma: not shippable as a second eval set.")
    elif (out["noise_as_pct_of_band"]["finetune"] or 100) > 15:
        print("\n=> separated, but fine-tune noise is still high; shippable only "
              "with that limit documented.")
    else:
        print("\n=> bbbp is shippable as a second eval set.")
