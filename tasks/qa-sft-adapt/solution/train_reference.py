"""Oracle solution for mleval/qa-sft-adapt: the recipe that set `reference`.

The same recipe as the `sft_full` arm of `spike/posttrain/sft_ladder.py`, run
against the training file the agent gets. Running it on the shipped data
reproduces the `reference_acc` in tests/grader/private/anchors.json to within
seed noise; `spike/posttrain/results/qa_anchors.json` records the seeds it was
measured over.

Supervised fine-tuning, in its plainest form: render each item as
`Question: ...\nAnswer: <correct answer>`, take the loss on the answer tokens
only, early-stop on a validation slice carved out of the agent's own file.

Deliberately ordinary. It does not: search the learning rate, mix in the
distractors as negatives, calibrate for answer length, use a chat template,
ensemble seeds, or train past the point where a 4-hour CPU budget stops being
comfortable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_LEN = 160


def render_prompt(question: str) -> str:
    """Must match the verifier's rendering exactly."""
    return f"Question: {question.strip()}\nAnswer:"


def render_choice(choice: str) -> str:
    return f" {choice.strip()}"


def pack(tok, pairs):
    """(input_ids, attention_mask, labels) with the prompt masked out of the loss."""
    ids, labels = [], []
    for prompt, cont in pairs:
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(cont, add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
        seq = (p + c)[-MAX_LEN:]
        n_c = min(len(c), len(seq))
        ids.append(seq)
        labels.append([-100] * (len(seq) - n_c) + seq[len(seq) - n_c:])
    width = max(len(s) for s in ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    inp = torch.full((len(ids), width), pad, dtype=torch.long)
    lab = torch.full((len(ids), width), -100, dtype=torch.long)
    att = torch.zeros((len(ids), width), dtype=torch.long)
    for i, s in enumerate(ids):
        inp[i, :len(s)] = torch.tensor(s)
        lab[i, :len(s)] = torch.tensor(labels[i])
        att[i, :len(s)] = 1
    return inp, att, lab


def choice_accuracy(model, tok, df, bs: int = 32) -> float:
    """The verifier's metric, recomputed here for early stopping."""
    model.eval()
    qs = df["question"].astype(str).tolist()
    chs = [json.loads(c) for c in df["choices"]]
    gold = df["answer_idx"].to_numpy()

    flat, owner = [], []
    for qi, (q, cs) in enumerate(zip(qs, chs)):
        for ch in cs:
            flat.append((render_prompt(q), render_choice(ch)))
            owner.append(qi)

    scores = np.full(len(flat), -np.inf)
    with torch.no_grad():
        for i in range(0, len(flat), bs):
            batch = flat[i:i + bs]
            inp, att, lab = pack(tok, batch)
            logits = model(input_ids=inp, attention_mask=att).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            for j in range(len(batch)):
                keep = (lab[j] != -100).nonzero().squeeze(-1)
                st, en = int(keep[0]), int(keep[-1])
                tgt = inp[j, st:en + 1]
                lp = logprobs[j, st - 1:en, :].gather(1, tgt.unsqueeze(1)).squeeze(1)
                scores[i + j] = float(lp.mean())
    model.train()

    per_q: list[list[float]] = [[] for _ in qs]
    for k, qi in enumerate(owner):
        per_q[qi].append(scores[k])
    pred = np.array([int(np.argmax(s)) for s in per_q])
    return float((pred == gold).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/app/data/qa_train.csv")
    ap.add_argument("--base", default="/app/base_model")
    ap.add_argument("--out", default="/app/final_model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.data)
    print(f"training items: {len(df):,}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base)

    pairs = []
    for _, r in df.iterrows():
        gold = json.loads(r["choices"])[int(r["answer_idx"])]
        pairs.append((render_prompt(str(r["question"])), render_choice(str(gold))))

    order = rng.permutation(len(pairs))
    n_val = max(1, int(0.1 * len(order)))
    val_idx, fit_idx = order[:n_val], order[n_val:]
    val_df = df.iloc[val_idx].reset_index(drop=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * max(1, len(fit_idx) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.1,
        anneal_strategy="linear")

    best, best_state, best_ep, t0, step = -1.0, None, -1, time.time(), 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(fit_idx))
        for i in range(0, len(fit_idx), args.bs):
            j = [fit_idx[k] for k in perm[i:i + args.bs]]
            inp, att, lab = pack(tok, [pairs[k] for k in j])
            loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
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
        v = choice_accuracy(model, tok, val_df)
        print(f"epoch {ep+1}/{args.epochs} val_acc={v:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(json.dumps({"best_epoch": best_ep + 1, "best_val_acc": round(best, 4),
                      "seconds": round(time.time() - t0, 1), "n_train": len(df)}),
          flush=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
