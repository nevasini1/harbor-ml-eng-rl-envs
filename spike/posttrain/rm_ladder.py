"""Effort ladder for the preference / reward-model track.

Same question the mol track asked, asked of RLHF stage 2: **do the pretrained
weights do any work, and is the gap between "no adaptation" and "real adaptation"
large compared with seed noise?**

Four arms, in increasing order of engineering:

  random_init   - identical architecture, randomly initialized, fully trained.
                  If this matches the pretrained arms the task measures nothing,
                  which is exactly how the Hydro track died.
  frozen_probe  - mean-pooled frozen embeddings, Bradley-Terry logistic model on
                  the (chosen - rejected) feature difference. ~1 minute of work.
  frozen_head   - frozen encoder, trained MLP head on the same embeddings,
                  early-stopped on a validation slice.
  finetune      - full fine-tune under the pairwise ranking loss.

`base` for the reward is the *ceiling* of {frozen_probe, frozen_head}, not either
one by name: the mol track twice found the two rungs swapping places between eval
sets, and pinning to the loser pays every lazy submission a slice of the reward.

Metric: pairwise accuracy - the fraction of held-out pairs where the model scores
the human-preferred response above the other. Chance is exactly 0.5, which makes
a reward band interpretable without further calibration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
# Set from the command line; screening compares candidate bases and truncation
# lengths against each other, and both change the anchors, so neither can be a
# constant baked into the file that measures them.
BASE_MODEL = "distilroberta-base"
MAX_LEN = 256

# Which hh subsets feed which eval set. Training draws from the same two
# collections the in-distribution eval sets come from; `online` is a different
# collection round and is screened as a shift set.
EVAL_SUBSETS = {
    "helpful": ["helpful-base"],
    "harmless": ["harmless-base"],
    "online": ["helpful-online"],
}
TRAIN_SUBSETS = ["helpful-base", "harmless-base"]


def private_seed() -> int:
    return int((HERE / "PRIVATE_SEED").read_text().strip())


# ------------------------------------------------------------------- splits

def load_split(split_dir: Path, eval_sets: list[str]):
    """Read the locked split written by make_splits.py.

    Anchors must be measured on the split that ships, not on a fresh draw: the
    mol track learned twice that base and reference move when either the split or
    the training size changes, which is why the ladder reads files here rather
    than re-partitioning.
    """
    train = pd.read_csv(split_dir / "agent" / "hh_train.csv.gz")
    tests = {n: pd.read_csv(split_dir / "private" / f"{n}_test.csv")
             for n in eval_sets}
    return train, tests


def build_split(n_train: int, n_test: int, eval_sets: list[str], seed: int):
    """Prompt-disjoint split. Whole prompts move together, never rows.

    A prompt appears in several pairs (different response samplings), so
    splitting on rows would put the same context on both sides and let a model
    score by memorising contexts rather than by judging responses.
    """
    df = pd.read_csv(DATA / "hh_pairs.csv.gz")
    rng = np.random.default_rng(seed)

    prompts = df["prompt"].astype(str)
    uniq = pd.Index(prompts.unique())
    # A prompt can in principle appear under two subsets; assign it to one.
    held = {}
    test_frames = {}
    for name in eval_sets:
        pool = df[df["subset"].isin(EVAL_SUBSETS[name])]
        cand = pd.Index(pool["prompt"].astype(str).unique())
        cand = cand.difference(pd.Index(list(held)))
        take = rng.permutation(len(cand))[: max(n_test * 3, 1)]
        chosen_prompts = set(cand[take])
        rows = pool[pool["prompt"].astype(str).isin(chosen_prompts)]
        rows = rows.iloc[rng.permutation(len(rows))[:n_test]].reset_index(drop=True)
        test_frames[name] = rows
        held |= {p: name for p in chosen_prompts}

    train_pool = df[df["subset"].isin(TRAIN_SUBSETS)]
    train_pool = train_pool[~train_pool["prompt"].astype(str).isin(set(held))]
    idx = rng.permutation(len(train_pool))[:n_train]
    train = train_pool.iloc[idx].reset_index(drop=True)

    assert set(train["prompt"]) & set(held) == set(), "prompt leaked into train"
    del uniq, prompts
    return train, test_frames


# --------------------------------------------------------------- featurizing

def texts_of(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    p = df["prompt"].astype(str).tolist()
    return p, df["chosen"].astype(str).tolist(), df["rejected"].astype(str).tolist()


def render(prompt: str, response: str) -> str:
    """The one input format. The verifier uses this string and nothing else.

    Fixed here rather than left to the agent because the verifier scores a bare
    checkpoint: if two submissions expected different renderings, their scores
    would not be comparable and the anchors would mean nothing.
    """
    return f"{prompt}\n\nAssistant: {response}"


def encode(tok, prompts, responses, device):
    """Single segment, truncated from the left.

    Left truncation is what keeps the *response* intact -- it is what is being
    judged, and a policy that ate it would make pairs unscoreable by content.
    A sentence-pair encoding with `only_first` would have been more precise, but
    it raises outright when the response alone exceeds max_length, and 4% of
    hh-rlhf responses do.
    """
    tok.truncation_side = "left"
    enc = tok([render(p, r) for p, r in zip(prompts, responses)],
              return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_LEN)
    return {k: v.to(device) for k, v in enc.items()}


def embed(model, tok, prompts, responses, device, bs=32) -> np.ndarray:
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            enc = encode(tok, prompts[i:i + bs], responses[i:i + bs], device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.concatenate(out)


def embed_frame(model, tok, df, device) -> tuple[np.ndarray, np.ndarray]:
    p, c, r = texts_of(df)
    return embed(model, tok, p, c, device), embed(model, tok, p, r, device)


# ------------------------------------------------------------------ the arms

def arm_length_only(tests: dict) -> dict:
    """Predict the longer response. No model, no parameters, no training.

    This rung exists because it beat everything else on the first cut of this
    task: 0.6031 on the `helpful_base` holdout, against 0.6042 for a full
    fine-tune and 0.5646 for the frozen-encoder ceiling. An eval set where this
    scores far from 0.5 is measuring response length, and any anchor that does
    not include it is paying every submission for free.

    It stays in the ladder after the split was length-balanced, as the check that
    the balancing still holds -- it should read 0.5000 exactly.
    """
    out = {}
    for name, te in tests.items():
        lc = te["chosen"].astype(str).str.len()
        lr = te["rejected"].astype(str).str.len()
        out[name] = float(((lc > lr).sum() + 0.5 * (lc == lr).sum()) / len(te))
    return out


def arm_frozen_probe(feats, seed: int) -> dict:
    """Bradley-Terry logistic regression on the chosen-minus-rejected difference.

    Fitting on differences with no intercept is the linear reward model: a single
    weight vector w whose score w.x ranks responses. Both orderings are supplied
    so the fit cannot exploit a constant sign.
    """
    from sklearn.linear_model import LogisticRegression

    (Ctr, Rtr), tests = feats["train"], feats["tests"]
    X = np.concatenate([Ctr - Rtr, Rtr - Ctr])
    y = np.concatenate([np.ones(len(Ctr)), np.zeros(len(Ctr))])
    clf = LogisticRegression(max_iter=3000, fit_intercept=False, random_state=seed)
    clf.fit(X, y)
    w = clf.coef_.ravel()
    return {name: float(np.mean((C - R) @ w > 0)) for name, (C, R) in tests.items()}


def arm_frozen_head(feats, seed: int, epochs: int = 60) -> dict:
    """Trained MLP head on the same frozen embeddings, early-stopped on val."""
    import torch

    torch.manual_seed(seed)
    (Ctr, Rtr), tests = feats["train"], feats["tests"]
    n_val = max(1, int(0.1 * len(Ctr)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Ctr))
    va, fit = order[:n_val], order[n_val:]

    d = Ctr.shape[1]
    head = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.Tanh(),
                               torch.nn.Linear(256, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    C = torch.tensor(Ctr, dtype=torch.float32)
    R = torch.tensor(Rtr, dtype=torch.float32)

    best, best_state = -1.0, None
    bs = 64
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(fit))
        for i in range(0, len(fit), bs):
            idx = torch.tensor(fit[perm[i:i + bs].numpy()])
            loss = -torch.nn.functional.logsigmoid(
                head(C[idx]) - head(R[idx])).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
        head.eval()
        with torch.no_grad():
            acc = float(((head(C[va]) - head(R[va])) > 0).float().mean())
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    out = {}
    with torch.no_grad():
        for name, (Cte, Rte) in tests.items():
            s = head(torch.tensor(Cte, dtype=torch.float32)) - \
                head(torch.tensor(Rte, dtype=torch.float32))
            out[name] = float((s > 0).float().mean())
    return out | {"_val": best}


def arm_finetune(train, test_frames, seed: int, device: str, epochs: int,
                 lr: float, head_lr: float, bs: int, random_init: bool,
                 save_to: Path | None = None) -> dict:
    """Full fine-tune under the pairwise ranking loss; best-val epoch wins."""
    import torch
    from transformers import (AutoConfig, AutoModelForSequenceClassification,
                              AutoTokenizer)

    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if random_init:
        cfg = AutoConfig.from_pretrained(BASE_MODEL, num_labels=1)
        model = AutoModelForSequenceClassification.from_config(cfg)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=1)
    model.to(device)

    p, c, r = texts_of(train)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(p))
    n_val = max(1, int(0.1 * len(order)))
    va, fit = order[:n_val], order[n_val:]

    head = [q for n, q in model.named_parameters() if n.startswith("classifier")]
    body = [q for n, q in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": lr},
                             {"params": head, "lr": head_lr}], weight_decay=0.01)
    steps = epochs * max(1, len(fit) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr, head_lr], total_steps=steps, pct_start=0.1,
        anneal_strategy="linear")

    def pair_acc(idx):
        model.eval()
        hits, n = 0, 0
        with torch.no_grad():
            for i in range(0, len(idx), 32):
                j = idx[i:i + 32]
                sc = model(**encode(tok, [p[k] for k in j], [c[k] for k in j],
                                    device)).logits.squeeze(-1)
                sr = model(**encode(tok, [p[k] for k in j], [r[k] for k in j],
                                    device)).logits.squeeze(-1)
                hits += int((sc > sr).sum())
                n += len(j)
        return hits / max(n, 1)

    best, best_state, best_ep, t0, step = -1.0, None, -1, time.time(), 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(fit))
        for i in range(0, len(fit), bs):
            j = [fit[k] for k in perm[i:i + bs]]
            sc = model(**encode(tok, [p[k] for k in j], [c[k] for k in j],
                                device)).logits.squeeze(-1)
            sr = model(**encode(tok, [p[k] for k in j], [r[k] for k in j],
                                device)).logits.squeeze(-1)
            loss = -torch.nn.functional.logsigmoid(sc - sr).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step < steps:
                sched.step()
        v = pair_acc(list(va))
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        print(f"    ep {ep+1}/{epochs} val={v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    out = {}
    for name, te in test_frames.items():
        pp, cc, rr = texts_of(te)
        model.eval()
        hits = 0
        import torch as _t
        with _t.no_grad():
            for i in range(0, len(pp), 32):
                sc = model(**encode(tok, pp[i:i+32], cc[i:i+32], device)).logits.squeeze(-1)
                sr = model(**encode(tok, pp[i:i+32], rr[i:i+32], device)).logits.squeeze(-1)
                hits += int((sc > sr).sum())
        out[name] = hits / len(pp)
    if save_to is not None:
        save_to.mkdir(parents=True, exist_ok=True)
        model.to("cpu").save_pretrained(save_to)
        tok.save_pretrained(save_to)
    return out | {"_val": best, "_best_epoch": best_ep + 1,
                  "_seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------- main

def main() -> None:
    global BASE_MODEL, MAX_LEN
    import torch
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--eval-sets", default="helpful,harmless,online")
    ap.add_argument("--arms", default="frozen_probe,frozen_head,finetune,random_init")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/rm_screen.json")
    ap.add_argument("--save-oracle", default="")
    ap.add_argument("--split-dir", default="",
                    help="measure on the locked split instead of a fresh draw")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    args = ap.parse_args()
    BASE_MODEL, MAX_LEN = args.base_model, args.max_len

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    eval_sets = args.eval_sets.split(",")
    arms = args.arms.split(",")

    if args.split_dir:
        train, test_frames = load_split(Path(args.split_dir), eval_sets)
    else:
        train, test_frames = build_split(args.n_train, args.n_test, eval_sets,
                                         private_seed())
    print(f"train={len(train):,} pairs  " +
          "  ".join(f"{k}={len(v):,}" for k, v in test_frames.items()) +
          f"  device={device}")

    results: dict = {"config": vars(args) | {"device": device,
                                             "base_model": BASE_MODEL,
                                             "n_train": len(train)},
                     "arms": {}}

    if {"frozen_probe", "frozen_head"} & set(arms):
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        enc_model = AutoModel.from_pretrained(BASE_MODEL).to(device)
        t0 = time.time()
        feats = {"train": embed_frame(enc_model, tok, train, device),
                 "tests": {k: embed_frame(enc_model, tok, v, device)
                           for k, v in test_frames.items()}}
        print(f"  embedded in {time.time()-t0:.0f}s")
        del enc_model

    for arm in arms:
        per_seed = []
        for s in range(args.seeds):
            t0 = time.time()
            if arm == "frozen_probe":
                r = arm_frozen_probe(feats, s)
            elif arm == "frozen_head":
                r = arm_frozen_head(feats, s)
            elif arm in ("finetune", "random_init"):
                save = Path(args.save_oracle) if (args.save_oracle and
                                                  arm == "finetune" and s == 0) else None
                r = arm_finetune(train, test_frames, s, device, args.epochs,
                                 args.lr, args.head_lr, args.bs,
                                 random_init=(arm == "random_init"), save_to=save)
            else:
                raise SystemExit(f"unknown arm {arm}")
            per_seed.append(r)
            print(f"  {arm} seed {s}: " +
                  "  ".join(f"{k}={v:.4f}" for k, v in r.items()
                            if not k.startswith("_")) +
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        agg = {}
        for name in eval_sets:
            vals = [r[name] for r in per_seed]
            agg[name] = {"mean": round(float(np.mean(vals)), 4),
                         "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1
                                      else 0.0, 4),
                         "seeds": [round(v, 4) for v in vals]}
        results["arms"][arm] = agg | {
            "detail": [{k: v for k, v in r.items() if k.startswith("_")}
                       for r in per_seed]}

    dest = HERE / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2))

    print(f"\n{'arm':<14}" + "".join(f"{n:>12}" for n in eval_sets))
    for arm, agg in results["arms"].items():
        print(f"{arm:<14}" + "".join(f"{agg[n]['mean']:>12.4f}" for n in eval_sets))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
