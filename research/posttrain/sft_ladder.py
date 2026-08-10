"""Effort ladder for the supervised-fine-tuning track.

A small causal LM is post-trained on multiple-choice QA and scored the way an
LM eval harness scores one: each candidate answer is appended to a fixed prompt,
the model's mean log-probability over the answer tokens is read off, and the
highest-scoring candidate is the prediction. No generation, no sampling, no
judge -- the metric is a deterministic function of the weights, which is what
makes it usable as a reward.

Arms, in increasing order of engineering:

  zero_shot   - the provided checkpoint, untouched. This is what an agent that
                does nothing submits, so it is the hard floor.
  head_only   - train only the output embedding matrix; the transformer body
                never moves. The cheap "adaptation" that is not adaptation.
  sft_full    - supervised fine-tuning of the whole model on (question, correct
                answer) pairs, loss on the answer tokens only.
  random_init - identical architecture, randomly initialized, same SFT. If this
                lands near the pretrained arms, the task is measuring format
                compliance rather than knowledge, and it is worthless.

`base` for the reward is the ceiling of {zero_shot, head_only}, not either one by
name -- the same rule the mol track had to learn twice.
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
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
MAX_LEN = 160


def private_seed() -> int:
    return int((HERE / "PRIVATE_SEED").read_text().strip())


# ---------------------------------------------------------------- rendering

def render_prompt(question: str) -> str:
    """The one prompt format. The verifier uses this string and nothing else."""
    return f"Question: {question.strip()}\nAnswer:"


def render_choice(choice: str) -> str:
    return f" {choice.strip()}"


# ------------------------------------------------------------------- splits

def load_split(split_dir: Path, sources: list[str]):
    """Read the locked split written by make_splits.py.

    Anchors must be measured on the split that ships, not on a fresh draw.
    """
    train = pd.read_csv(split_dir / "agent" / "qa_train.csv")
    tests = {n: pd.read_csv(split_dir / "private" / f"{n}_test.csv")
             for n in sources}
    return train, tests, 0


def build_split(n_train: int, n_test: int, sources: list[str], seed: int):
    """Private holdout per source; training data is the pooled remainder.

    Splitting is on items, not on questions-with-shared-stems, because these
    corpora contain no repeated stems: `qa_items.csv.gz` has one row per item and
    duplicate questions are rare enough to check rather than engineer around.
    """
    df = pd.read_csv(DATA / "qa_items.csv.gz")
    df = df[df["source"].isin(sources)].reset_index(drop=True)
    rng = np.random.default_rng(seed)

    tests, held = {}, set()
    for name in sources:
        pool = df.index[df["source"] == name].to_numpy()
        take = pool[rng.permutation(len(pool))[:n_test]]
        tests[name] = df.loc[take].reset_index(drop=True)
        held |= set(take.tolist())

    rest = df.drop(index=sorted(held))
    idx = rng.permutation(len(rest))[:n_train]
    train = rest.iloc[idx].reset_index(drop=True)

    dupes = set(train["question"]) & {q for te in tests.values() for q in te["question"]}
    if dupes:
        train = train[~train["question"].isin(dupes)].reset_index(drop=True)
    return train, tests, len(dupes)


# ------------------------------------------------------------------ scoring

def choice_scores(model, tok, questions, choices_per_q, device, bs: int = 32):
    """Mean log-probability per answer token, for every candidate answer.

    Length normalization is by token count. Without it the score is dominated by
    answer length -- a one-word distractor outranks the correct four-word answer
    on almost every item, which measures brevity rather than knowledge.
    """
    import torch

    flat, owner = [], []
    for qi, (q, chs) in enumerate(zip(questions, choices_per_q)):
        for ch in chs:
            flat.append((render_prompt(q), render_choice(ch)))
            owner.append(qi)

    out = np.full(len(flat), -np.inf)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(flat), bs):
            batch = flat[i:i + bs]
            ids, starts = [], []
            for prompt, cont in batch:
                p = tok(prompt, add_special_tokens=False)["input_ids"]
                c = tok(cont, add_special_tokens=False)["input_ids"]
                if not c:
                    c = [tok.eos_token_id]
                seq = (p + c)[-MAX_LEN:]
                starts.append(max(len(seq) - len(c), 1))
                ids.append(seq)
            width = max(len(s) for s in ids)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            inp = torch.full((len(ids), width), pad, dtype=torch.long)
            att = torch.zeros((len(ids), width), dtype=torch.long)
            for j, s in enumerate(ids):
                inp[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            logits = model(input_ids=inp.to(device),
                           attention_mask=att.to(device)).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1).cpu()
            for j, s in enumerate(ids):
                st = starts[j]
                tgt = torch.tensor(s[st:])
                lp = logprobs[j, st - 1:len(s) - 1, :].gather(
                    1, tgt.unsqueeze(1)).squeeze(1)
                out[i + j] = float(lp.mean())

    per_q: list[list[float]] = [[] for _ in questions]
    for k, qi in enumerate(owner):
        per_q[qi].append(out[k])
    return per_q


def accuracy(model, tok, df: pd.DataFrame, device, bs: int = 32) -> float:
    qs = df["question"].astype(str).tolist()
    chs = [json.loads(c) for c in df["choices"]]
    gold = df["answer_idx"].to_numpy()
    scores = choice_scores(model, tok, qs, chs, device, bs)
    pred = np.array([int(np.argmax(s)) for s in scores])
    return float((pred == gold).mean())


# --------------------------------------------------------------------- arms

def sft(train: pd.DataFrame, tests: dict, seed: int, device: str, epochs: int,
        lr: float, bs: int, mode: str, save_to: Path | None = None) -> dict:
    """mode: 'full' | 'head' | 'random_init'."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if mode == "random_init":
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(BASE_MODEL))
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.to(device)

    if mode == "head":
        # Output embeddings only. On a tied-embedding checkpoint this is also the
        # input embedding matrix, which is the most generous reading of
        # "train only the head" -- deliberately so, since `base` must be the
        # ceiling of the no-body-adaptation class rather than its cheapest member.
        keep = set()
        for p in model.get_output_embeddings().parameters():
            keep.add(id(p))
        for p in model.parameters():
            p.requires_grad_(id(p) in keep)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    rows = train.reset_index(drop=True)
    texts, prompts = [], []
    for _, r in rows.iterrows():
        gold = json.loads(r["choices"])[int(r["answer_idx"])]
        prompts.append(render_prompt(str(r["question"])))
        texts.append(render_choice(str(gold)))

    order = rng.permutation(len(texts))
    n_val = max(1, int(0.1 * len(order)))
    val_idx, fit_idx = order[:n_val], order[n_val:]
    val_df = rows.iloc[val_idx].reset_index(drop=True)

    steps = epochs * max(1, len(fit_idx) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="linear")

    best, best_state, best_ep, t0, step = -1.0, None, -1, time.time(), 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(fit_idx))
        for i in range(0, len(fit_idx), bs):
            j = [fit_idx[k] for k in perm[i:i + bs]]
            ids, labels = [], []
            for k in j:
                p = tok(prompts[k], add_special_tokens=False)["input_ids"]
                c = tok(texts[k], add_special_tokens=False)["input_ids"] or \
                    [tok.eos_token_id]
                seq = (p + c)[-MAX_LEN:]
                # Loss on the answer tokens only: the question is context the
                # model is never asked to produce, and training on it spends
                # capacity on the wrong distribution.
                lab = [-100] * (len(seq) - len(c)) + seq[len(seq) - len(c):]
                ids.append(seq)
                labels.append(lab)
            width = max(len(s) for s in ids)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            inp = torch.full((len(ids), width), pad, dtype=torch.long)
            lb = torch.full((len(ids), width), -100, dtype=torch.long)
            att = torch.zeros((len(ids), width), dtype=torch.long)
            for m, s in enumerate(ids):
                inp[m, :len(s)] = torch.tensor(s)
                lb[m, :len(s)] = torch.tensor(labels[m])
                att[m, :len(s)] = 1
            loss = model(input_ids=inp.to(device), attention_mask=att.to(device),
                         labels=lb.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step < steps:
                sched.step()
        v = accuracy(model, tok, val_df, device)
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        print(f"    ep {ep+1}/{epochs} val={v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    out = {name: accuracy(model, tok, te, device) for name, te in tests.items()}
    if save_to is not None:
        save_to.mkdir(parents=True, exist_ok=True)
        model.to("cpu").save_pretrained(save_to)
        tok.save_pretrained(save_to)
    return out | {"_val": best, "_best_epoch": best_ep + 1,
                  "_seconds": round(time.time() - t0, 1)}


def zero_shot(tests: dict, device: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    return {name: accuracy(model, tok, te, device) for name, te in tests.items()}


# --------------------------------------------------------------------- main

def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--sources", default="arc_easy,sciq,openbookqa")
    ap.add_argument("--arms", default="zero_shot,head_only,sft_full,random_init")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/sft_screen.json")
    ap.add_argument("--save-oracle", default="")
    ap.add_argument("--split-dir", default="",
                    help="measure on the locked split instead of a fresh draw")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    sources = args.sources.split(",")

    if args.split_dir:
        train, tests, dupes = load_split(Path(args.split_dir), sources)
    else:
        train, tests, dupes = build_split(args.n_train, args.n_test, sources,
                                          private_seed())
    print(f"train={len(train):,}  " +
          "  ".join(f"{k}={len(v):,}" for k, v in tests.items()) +
          f"  (dropped {dupes} duplicate stems)  device={device}")

    results = {"config": vars(args) | {"device": device, "base_model": BASE_MODEL,
                                       "n_train": len(train)}, "arms": {}}
    for arm in args.arms.split(","):
        per_seed = []
        for s in range(args.seeds):
            t0 = time.time()
            if arm == "zero_shot":
                r = zero_shot(tests, device)
            elif arm == "head_only":
                r = sft(train, tests, s, device, args.epochs, 1e-3, args.bs, "head")
            elif arm in ("sft_full", "random_init"):
                save = Path(args.save_oracle) if (args.save_oracle and
                                                  arm == "sft_full" and s == 0) else None
                r = sft(train, tests, s, device, args.epochs, args.lr, args.bs,
                        "full" if arm == "sft_full" else "random_init", save)
            else:
                raise SystemExit(f"unknown arm {arm}")
            per_seed.append(r)
            print(f"  {arm} seed {s}: " +
                  "  ".join(f"{k}={v:.4f}" for k, v in r.items()
                            if not k.startswith("_")) +
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            if arm == "zero_shot":
                break  # deterministic; seeds would be identical
        agg = {}
        for name in sources:
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
    print(f"\n{'arm':<14}" + "".join(f"{n:>12}" for n in sources))
    for arm, agg in results["arms"].items():
        print(f"{arm:<14}" + "".join(f"{agg[n]['mean']:>12.4f}" for n in sources))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
