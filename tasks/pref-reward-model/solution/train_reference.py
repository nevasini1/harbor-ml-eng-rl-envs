"""Oracle solution for mleval/pref-reward-model: the recipe that set `reference`.

This is the *same* recipe as the `finetune` arm of
`research/posttrain/rm_ladder.py`, run against the same training file the agent
gets. Keeping one recipe in two places is a known hazard -- the mol task shipped
a reference script that did not reproduce the anchor it claimed, and the README
still carries that as an open question -- so the invariant here is a measured
number rather than a promise.

Measured: run through Harbor on Modal (`jobs/rm-oracle-modal/`), this script
scored **0.618915** on the held-out set, against a `reference_acc` of 0.6268.
That is -1.29 sigma on the reference arm's seed spread (0.0061) and 0.0008 below
the lowest of the five seeds that set the anchor.

Which is the expected shape, not a defect, and the reason is worth stating: the
anchor is a five-seed **mean**, so a single-seed run of the same recipe lands
below it about half the time by construction. An oracle trial should therefore be
read as "recovery near, and often under, 1.0" -- this one returned 0.5828. If you
want an oracle that reliably clears its own reference you have to either anchor on
a lower quantile than the mean or run the oracle multi-seed, and both change what
`reference` means.

It is deliberately a competent-but-ordinary fine-tune rather than a maximal one,
so that a strong agent can exceed it and earn the full reward. What it does not
do: tune the learning rate, use the response-length prior, ensemble seeds, sample
hard pairs, or train longer than the budget comfortably allows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 256


def render(prompt: str, response: str) -> str:
    """Must match the verifier's rendering exactly."""
    return f"{prompt}\n\nAssistant: {response}"


def encode(tok, prompts, responses, device):
    tok.truncation_side = "left"
    enc = tok([render(p, r) for p, r in zip(prompts, responses)],
              return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_LEN)
    return {k: v.to(device) for k, v in enc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/app/data/hh_train.csv.gz")
    ap.add_argument("--base", default="/app/base_model")
    ap.add_argument("--out", default="/app/final_model")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # This task has a GPU and the reference needs it: 2 epochs over 38,420 pairs
    # is ~13 minutes on an A10G and 6.7 hours on 8 CPU cores, against a 4-hour
    # budget. Falling back silently to CPU is the failure this script had on its
    # first Harbor run -- it does not error, it just never finishes.
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device visible; this recipe does not fit the "
              "time budget on CPU", flush=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.data)
    p = df["prompt"].astype(str).tolist()
    c = df["chosen"].astype(str).tolist()
    r = df["rejected"].astype(str).tolist()
    print(f"training pairs: {len(p):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1)
    model.to(device)
    print(f"device: {device}", flush=True)

    # Validation is carved out of the agent's own training file. The held-out
    # split is not in this container, so early stopping has to be done against
    # data the agent already has -- which is the same constraint the agent is
    # under, deliberately.
    order = rng.permutation(len(p))
    n_val = max(1, int(0.1 * len(order)))
    val_idx, fit_idx = order[:n_val], order[n_val:]

    head = [q for n, q in model.named_parameters() if n.startswith("classifier")]
    body = [q for n, q in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = args.epochs * max(1, len(fit_idx) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.1,
        anneal_strategy="linear")

    def pair_acc(idx):
        model.eval()
        hits = 0
        with torch.no_grad():
            for i in range(0, len(idx), 32):
                j = idx[i:i + 32]
                sc = model(**encode(tok, [p[k] for k in j],
                                    [c[k] for k in j], device)).logits.squeeze(-1)
                sr = model(**encode(tok, [p[k] for k in j],
                                    [r[k] for k in j], device)).logits.squeeze(-1)
                hits += int((sc > sr).sum())
        model.train()
        return hits / max(len(idx), 1)

    best, best_state, best_ep, t0, step = -1.0, None, -1, time.time(), 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(fit_idx))
        for i in range(0, len(fit_idx), args.bs):
            j = [fit_idx[k] for k in perm[i:i + args.bs]]
            sc = model(**encode(tok, [p[k] for k in j],
                                [c[k] for k in j], device)).logits.squeeze(-1)
            sr = model(**encode(tok, [p[k] for k in j],
                                [r[k] for k in j], device)).logits.squeeze(-1)
            # Bradley-Terry: maximize the log-odds that the preferred response
            # scores higher. No absolute target, because a reward model only has
            # to be right about ordering.
            loss = -torch.nn.functional.logsigmoid(sc - sr).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step < steps:
                sched.step()
            if step % 50 == 0:
                print(f"  step {step}/{steps} loss={float(loss):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        v = pair_acc(list(val_idx))
        print(f"epoch {ep+1}/{args.epochs} val_pairwise_acc={v:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    # save_pretrained writes whatever device the tensors are on; move to CPU so
    # the verifier -- which has no GPU -- can load the checkpoint.
    model.to("cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(json.dumps({"best_epoch": best_ep + 1, "best_val_pairwise_acc": round(best, 4),
                      "seconds": round(time.time() - t0, 1), "n_train": len(p),
                      "device": device}),
          flush=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
