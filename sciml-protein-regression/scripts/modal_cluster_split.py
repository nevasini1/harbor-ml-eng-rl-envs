"""Does a cluster-level split restore headroom? Measured, both arms, both splits.

The problem
-----------
On the shipped split the task is inverted: a frozen probe scores 0.546 +/- 0.005
(n=8) while a standard fine-tune scores 0.5169 +/- 0.005 (n=4). Doing the
intended work is reliably *worse* than not doing it, and t_strong (0.45) sits
below both. No threshold placement repairs that -- if there is no gap, there is
nothing for thresholds to separate.

The hypothesis
--------------
The shipped split buckets on sha256(seed + sequence), i.e. by *exact sequence*.
FLIP2 meltome-mixed is cross-species, so orthologs and close homologs of test
proteins sit in the agent's training file by construction: 89 of 3,421 private
test sequences share their first 60 residues with a training sequence. Frozen
ESM-2 embeddings are excellent at near-neighbour retrieval, so an easy split
flatters the probe. A split that separates *clusters* rather than sequences
should hurt the probe more than the fine-tune, restoring a usable gap.

If that is wrong -- if both arms fall together -- then the property itself does
not support this task at this model scale, and the honest conclusion is to change
the target rather than re-tune the bars.

Method
------
The pool is the shipped train + private test concatenated (that union *is* the
deduped public corpus). Embedding it once and caching means any split is an
indexing operation, so both splits are measured through identical code paths.

MMseqs2 clusters at 30% identity / 80% coverage. Clusters are assigned whole to
test, so no cluster straddles the boundary. Test size is matched to the current
3,427 so Spearman is computed on a comparably sized set.

Both arms use identical discipline on both splits:
    fit on train, early-stop on val, report on test. Test never informs training.

Run:  modal run sciml-protein-regression/scripts/modal_cluster_split.py
"""

from __future__ import annotations

import modal

TASK = "sciml-protein-regression"
BASE = "facebook/esm2_t6_8M_UR50D"
REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"

N_FROZEN_SEEDS = 5
N_FINETUNE_SEEDS = 3
MIN_SEQ_ID = 0.3
COVERAGE = 0.8

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar")
    # Static MMseqs2 build: the Debian package lags and the release binary is the
    # reference implementation everyone cites for identity-based clustering.
    .run_commands(
        "wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/m.tar.gz",
        "tar xzf /tmp/m.tar.gz -C /opt && rm /tmp/m.tar.gz",
    )
    .env({"PATH": "/opt/mmseqs/bin:/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/cache/hf"})
    .pip_install(
        "torch==2.6.0", "transformers==4.49.0", "scikit-learn==1.5.2",
        "pandas==2.2.3", "numpy==1.26.4", "scipy==1.14.1",
    )
    .add_local_file(f"{TASK}/environment/data/train.csv.gz", "/data/train.csv.gz")
    .add_local_file(f"{TASK}/tests/private_test/test.csv.gz", "/data/test.csv.gz")
)

cache = modal.Volume.from_name("esm-probe-cache", create_if_missing=True)
app = modal.App("esm-cluster-split")


def _pool():
    """The deduped corpus: shipped train+val plus the private test set."""
    import gzip

    import pandas as pd

    with gzip.open("/data/train.csv.gz", "rt") as fh:
        tr = pd.read_csv(fh)
    with gzip.open("/data/test.csv.gz", "rt") as fh:
        te = pd.read_csv(fh)
    tr["orig"] = tr["split"]          # "train" | "val"
    te["orig"] = "test"
    pool = pd.concat([tr, te], ignore_index=True)
    pool = pool.drop_duplicates("sequence").reset_index(drop=True)
    return pool


@app.function(image=image, timeout=3600, cpu=8, volumes={"/cache": cache})
def build_cluster_split(seed: int = 0) -> dict:
    """Cluster the pool at 30% identity and assign whole clusters to test."""
    import json
    import subprocess
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    pool = _pool()
    fa = Path("/tmp/pool.fasta")
    with fa.open("w") as fh:
        for i, s in enumerate(pool["sequence"].astype(str)):
            fh.write(f">{i}\n{s}\n")

    subprocess.run(
        ["mmseqs", "easy-cluster", str(fa), "/tmp/clu", "/tmp/tmp",
         "--min-seq-id", str(MIN_SEQ_ID), "-c", str(COVERAGE), "--cov-mode", "0",
         "-v", "1"],
        check=True, capture_output=True, text=True,
    )

    members = defaultdict(list)
    for line in Path("/tmp/clu_cluster.tsv").read_text().splitlines():
        rep, mem = line.split("\t")
        members[rep].append(int(mem))
    clusters = list(members.values())
    print(f"pool={len(pool)} clusters={len(clusters)} "
          f"largest={max(len(c) for c in clusters)}", flush=True)

    target = 3427                      # match the shipped private test size
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clusters))
    test_idx: list[int] = []
    for ci in order:
        if len(test_idx) >= target:
            break
        test_idx.extend(clusters[ci])
    test_set = set(test_idx)

    rest = [i for i in range(len(pool)) if i not in test_set]
    rest_clusters = [c for c in clusters if not (set(c) & test_set)]
    rng.shuffle(rest_clusters)
    val_idx: list[int] = []
    for c in rest_clusters:                       # val is cluster-disjoint too
        if len(val_idx) >= 1991:
            break
        val_idx.extend(c)
    val_set = set(val_idx)
    train_idx = [i for i in rest if i not in val_set]

    split = {"train": train_idx, "val": val_idx, "test": test_idx}
    Path("/cache/cluster_split.json").write_text(json.dumps(split))
    cache.commit()

    # How much near-duplicate leakage did the shipped split have, and this one?
    def leak(a_idx, b_idx) -> int:
        pref = {pool["sequence"][i][:60] for i in a_idx}
        return sum(1 for i in b_idx if pool["sequence"][i][:60] in pref)

    cur_train = pool.index[pool["orig"].isin(["train", "val"])].tolist()
    cur_test = pool.index[pool["orig"] == "test"].tolist()
    out = {
        "n_clusters": len(clusters),
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "shipped_split_prefix_leak": leak(cur_train, cur_test),
        "cluster_split_prefix_leak": leak(train_idx, test_idx),
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/cache": cache})
def embed_pool() -> str:
    """Embed the whole pool once (final layer, CLS + mean). Splits then index it."""
    import os

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = "/cache/pool_emb.npz"
    if os.path.exists(path):
        print("cache hit", flush=True)
        return path

    pool = _pool()
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = AutoModel.from_pretrained(BASE, revision=REVISION).to("cuda").eval()
    seqs = pool["sequence"].astype(str).tolist()

    cls_o, mean_o = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), 64):
            chunk = seqs[i : i + 64]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to("cuda")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            m[:, 0] = 0
            for j, s in enumerate(chunk):
                m[j, min(len(s) + 1, m.shape[1] - 1)] = 0
            cls_o.append(h[:, 0].float().cpu())
            mean_o.append(((h * m).sum(1) / m.sum(1).clamp_min(1e-6)).float().cpu())
            if i % 5120 == 0:
                print(f"  {i}/{len(seqs)}", flush=True)

    np.savez(path, cls=torch.cat(cls_o).numpy(), mean=torch.cat(mean_o).numpy())
    cache.commit()
    return path


def _split_indices(which: str):
    import json

    import numpy as np

    pool = _pool()
    if which == "shipped":
        tr = pool.index[pool["orig"] == "train"].to_numpy()
        va = pool.index[pool["orig"] == "val"].to_numpy()
        te = pool.index[pool["orig"] == "test"].to_numpy()
    else:
        s = json.loads(open("/cache/cluster_split.json").read())
        tr, va, te = (np.array(s["train"]), np.array(s["val"]), np.array(s["test"]))
    y = pool["target"].to_numpy(float)
    return tr, va, te, y, pool


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/cache": cache})
def frozen_arm(which: str, seed: int) -> dict:
    import numpy as np
    import torch
    from scipy.stats import spearmanr

    z = np.load("/cache/pool_emb.npz")
    tr, va, te, y, _ = _split_indices(which)
    mu, sd = float(y[tr].mean()), float(y[tr].std())

    best = {"private": -1.0}
    for pooling in ("cls", "mean"):
        E = z[pooling]
        Xtr = torch.tensor(E[tr], dtype=torch.float32, device="cuda")
        Xva = torch.tensor(E[va], dtype=torch.float32, device="cuda")
        Xte = torch.tensor(E[te], dtype=torch.float32, device="cuda")
        ttr = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32, device="cuda")

        torch.manual_seed(seed)
        hid = Xtr.shape[1]
        head = torch.nn.Sequential(
            torch.nn.Linear(hid, hid), torch.nn.Tanh(), torch.nn.Linear(hid, 1)
        ).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
        g = torch.Generator().manual_seed(seed)
        best_val, best_te, patience = -1.0, None, 0
        for _ in range(300):
            head.train()
            perm = torch.randperm(Xtr.shape[0], generator=g).cuda()
            for i in range(0, len(perm), 256):
                idx = perm[i : i + 256]
                opt.zero_grad()
                torch.nn.functional.mse_loss(
                    head(Xtr[idx]).squeeze(-1), ttr[idx]
                ).backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                rv = float(spearmanr(head(Xva).squeeze(-1).cpu().numpy(), y[va]).statistic)
            if rv > best_val + 1e-5:
                best_val, patience = rv, 0
                with torch.no_grad():
                    best_te = head(Xte).squeeze(-1).cpu().numpy()
            else:
                patience += 1
                if patience >= 15:
                    break
        r = float(spearmanr(best_te, y[te]).statistic)
        if r > best["private"]:
            best = {"private": r, "val": best_val, "pooling": pooling}
    print(f"[{which}] frozen seed {seed} -> {best}", flush=True)
    return {"split": which, "arm": "frozen", "seed": seed, **best}


@app.function(gpu="A10G", image=image, timeout=5400, volumes={"/cache": cache})
def finetune_arm(which: str, seed: int) -> dict:
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer, EsmForSequenceClassification

    tr, va, te, y, pool = _split_indices(which)
    seqs = pool["sequence"].astype(str).tolist()
    mu, sd = float(y[tr].mean()), float(y[tr].std())

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = EsmForSequenceClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=1, problem_type="regression"
    ).cuda()
    n_layers = model.config.num_hidden_layers
    for name, p in model.named_parameters():
        p.requires_grad = "classifier" in name or any(
            f"layer.{i}." in name for i in range(n_layers - 2, n_layers)
        )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-5, weight_decay=0.01
    )

    def ev(idx, crop=512, bs=64):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                enc = tok([seqs[k] for k in idx[i : i + bs]], return_tensors="pt",
                          padding=True, truncation=True, max_length=crop).to("cuda")
                preds.append(model(**enc).logits.squeeze(-1).float().cpu())
        return float(spearmanr(torch.cat(preds).numpy(), y[idx]).statistic)

    g = np.random.default_rng(seed)
    tgt = (y - mu) / sd
    best_val, best_te = -1.0, -1.0
    for epoch, crop in enumerate((256, 384, 512), start=1):
        model.train()
        order = g.permutation(tr)
        for i in range(0, len(order), 16):
            idx = order[i : i + 16]
            enc = tok([seqs[k] for k in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=crop).to("cuda")
            t = torch.tensor(tgt[idx], dtype=torch.float32, device="cuda")
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(**enc).logits.squeeze(-1), t).backward()
            opt.step()
        rv = ev(va)
        print(f"[{which}] ft seed {seed} epoch {epoch} val={rv:.4f}", flush=True)
        if rv > best_val:
            best_val, best_te = rv, ev(te)
    return {"split": which, "arm": "finetune", "seed": seed,
            "private": best_te, "val": best_val}


@app.local_entrypoint()
def main():
    import json
    import statistics as st
    from pathlib import Path

    print("clustering pool at 30% identity...")
    cluster_info = build_cluster_split.remote()
    print("embedding pool (cached after first run)...")
    embed_pool.remote()

    jobs = [(w, s) for w in ("shipped", "cluster30") for s in range(N_FROZEN_SEEDS)]
    ft_jobs = [(w, s) for w in ("shipped", "cluster30") for s in range(N_FINETUNE_SEEDS)]
    frozen = list(frozen_arm.starmap(jobs))
    finetuned = list(finetune_arm.starmap(ft_jobs))

    def stats(rows):
        v = [r["private"] for r in rows]
        return {"n": len(v), "mean": round(st.mean(v), 4),
                "std": round(st.stdev(v), 4) if len(v) > 1 else 0.0}

    out = {"cluster_info": cluster_info, "splits": {}}
    for w in ("shipped", "cluster30"):
        fz = stats([r for r in frozen if r["split"] == w])
        ft = stats([r for r in finetuned if r["split"] == w])
        gap = ft["mean"] - fz["mean"]
        pooled = (fz["std"] ** 2 + ft["std"] ** 2) ** 0.5
        out["splits"][w] = {
            "frozen": fz, "finetune": ft,
            "headroom": round(gap, 4),
            "headroom_sigma": round(gap / pooled, 2) if pooled else None,
        }
    out["raw"] = frozen + finetuned

    dest = Path(__file__).resolve().parent / "cluster_split.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print("\n" + "=" * 68)
    print(f"near-duplicate leak (shared 60-residue prefix, train->test):")
    print(f"  shipped split : {cluster_info['shipped_split_prefix_leak']}")
    print(f"  cluster30     : {cluster_info['cluster_split_prefix_leak']}")
    for w, d in out["splits"].items():
        print(f"\n{w}:")
        print(f"  frozen   {d['frozen']}")
        print(f"  finetune {d['finetune']}")
        print(f"  headroom {d['headroom']}  ({d['headroom_sigma']} sigma)")
    print(f"\nwrote {dest}")
