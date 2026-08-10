import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", default="/app/embeddings.npz")
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    frame = pd.read_csv("/data/train.csv.gz")
    if args.limit:
        frame = frame.iloc[: args.limit].copy()
    seqs = frame.sequence.tolist()
    tokenizer = AutoTokenizer.from_pretrained("/models/esm2_t6_8M_UR50D")
    model = AutoModel.from_pretrained("/models/esm2_t6_8M_UR50D")
    model.eval()

    n = len(seqs)
    layers = model.config.num_hidden_layers + 1
    cls = np.empty((n, layers, model.config.hidden_size), dtype=np.float32)
    mean = np.empty_like(cls)
    last_std = np.empty((n, model.config.hidden_size), dtype=np.float32)
    order = sorted(range(n), key=lambda i: len(seqs[i]))
    started = time.time()
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            inds = order[start : start + args.batch_size]
            batch = tokenizer(
                [seqs[i] for i in inds], padding=True, truncation=True,
                max_length=514, return_tensors="pt"
            )
            out = model(**batch, output_hidden_states=True)
            # Exclude BOS, EOS, and padding from residue pooling.
            lengths = batch.attention_mask.sum(1) - 2
            for j, original_i in enumerate(inds):
                length = int(lengths[j])
                for layer_i, hidden in enumerate(out.hidden_states):
                    values = hidden[j, 1 : length + 1]
                    cls[original_i, layer_i] = hidden[j, 0].numpy()
                    mean[original_i, layer_i] = values.mean(0).numpy()
                last_std[original_i] = out.last_hidden_state[j, 1 : length + 1].std(0).numpy()
            if start == 0 or (start // args.batch_size) % 100 == 0:
                elapsed = time.time() - started
                print(f"{start + len(inds)}/{n} elapsed={elapsed:.1f}s", flush=True)

    np.savez_compressed(
        args.output, cls=cls, mean=mean, last_std=last_std,
        target=frame.target.values.astype(np.float32),
        split=frame.split.values,
    )
    print(f"saved {args.output}; total {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
