import argparse
import copy
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE = "/models/esm2_t6_8M_UR50D"


def batches_for_epoch(indices, lengths, batch_size, seed):
    """Shuffle local length buckets, retaining random batches with little padding."""
    rng = random.Random(seed)
    ordered = sorted(indices, key=lambda i: lengths[i])
    buckets = [ordered[i : i + 256] for i in range(0, len(ordered), 256)]
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    flattened = [i for bucket in buckets for i in bucket]
    return [flattened[i : i + batch_size] for i in range(0, len(flattened), batch_size)]


def crop_batch(sequences, indices, max_residues, rng):
    result = []
    for i in indices:
        seq = sequences[i]
        if len(seq) > max_residues:
            # Include the inference-aligned N-terminal view often, while exposing
            # the model to the remainder of long proteins as well.
            if rng.random() < 0.35:
                start = 0
            else:
                start = rng.randrange(len(seq) - max_residues + 1)
            seq = seq[start : start + max_residues]
        result.append(seq)
    return result


@torch.inference_mode()
def evaluate(model, tokenizer, sequences, targets, indices, batch_size):
    model.eval()
    ordered = sorted(indices, key=lambda i: len(sequences[i]))
    predictions = []
    actual = []
    for start in range(0, len(ordered), batch_size):
        inds = ordered[start : start + batch_size]
        encoded = tokenizer([sequences[i] for i in inds], padding=True, return_tensors="pt")
        pred = model(**encoded).logits[:, 0].float().numpy()
        predictions.extend(pred.tolist())
        actual.extend(targets[inds].tolist())
    return float(spearmanr(actual, predictions).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="/app/frozen_head.pt")
    ap.add_argument("--output", default="/app/working_model")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--crop-schedule", default="192,256,320,384")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--unfreeze-layers", type=int, default=2)
    ap.add_argument("--encoder-lr", type=float, default=8e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = pd.read_csv("/data/train.csv.gz")
    sequences = frame.sequence.tolist()
    targets_raw = frame.target.values.astype(np.float32)
    train_indices = np.flatnonzero(frame.split.values == "train").tolist()
    val_indices = np.flatnonzero(frame.split.values == "val").tolist()
    lengths = [len(s) for s in sequences]
    head_checkpoint = torch.load(args.head, map_location="cpu", weights_only=True)
    y_mean = head_checkpoint["y_mean"]
    y_std = head_checkpoint["y_std"]
    targets = (targets_raw - y_mean) / y_std

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=1, ignore_mismatched_sizes=True
    )
    model.config.problem_type = "regression"
    model.config.id2label = {0: "melting_temperature"}
    model.config.label2id = {"melting_temperature": 0}
    model.classifier.dense.load_state_dict({
        "weight": head_checkpoint["state_dict"]["dense.weight"],
        "bias": head_checkpoint["state_dict"]["dense.bias"],
    })
    model.classifier.out_proj.load_state_dict({
        "weight": head_checkpoint["state_dict"]["out_proj.weight"],
        "bias": head_checkpoint["state_dict"]["out_proj.bias"],
    })

    for param in model.esm.parameters():
        param.requires_grad = False
    for layer in model.esm.encoder.layer[-args.unfreeze_layers :]:
        for param in layer.parameters():
            param.requires_grad = True
    # The final encoder normalization is cheap and task-relevant.
    if hasattr(model.esm.encoder, "emb_layer_norm_after"):
        for param in model.esm.encoder.emb_layer_norm_after.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    print(
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
        f"of {sum(p.numel() for p in model.parameters()):,}", flush=True
    )
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 0.001},
    ])
    steps_per_epoch = math.ceil(math.ceil(len(train_indices) / args.batch_size) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.06))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.08, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    initial = evaluate(model, tokenizer, sequences, targets, val_indices, 16)
    print(f"initial full-length val rho={initial:.6f}", flush=True)
    best_score = initial
    best_epoch = -1
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    crop_schedule = [int(x) for x in args.crop_schedule.split(",")]
    if len(crop_schedule) < args.epochs:
        crop_schedule.extend([crop_schedule[-1]] * (args.epochs - len(crop_schedule)))
    global_step = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        max_residues = crop_schedule[epoch]
        rng = random.Random(args.seed + 1009 * epoch)
        batches = batches_for_epoch(
            train_indices, [min(x, max_residues) for x in lengths],
            args.batch_size, args.seed + epoch
        )
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for batch_i, inds in enumerate(batches):
            strings = crop_batch(sequences, inds, max_residues, rng)
            encoded = tokenizer(strings, padding=True, return_tensors="pt")
            labels = torch.from_numpy(targets[inds].copy())
            pred = model(**encoded).logits[:, 0]
            regression = torch.nn.functional.smooth_l1_loss(pred, labels, beta=0.5)
            # A small pairwise term aligns optimization with rank correlation.
            differences = pred[:, None] - pred[None, :]
            truth_sign = torch.sign(labels[:, None] - labels[None, :])
            pairs = truth_sign != 0
            ranking = torch.nn.functional.softplus(-truth_sign[pairs] * differences[pairs]).mean()
            loss = (regression + 0.08 * ranking) / args.grad_accum
            loss.backward()
            running += float(regression.detach())
            if (batch_i + 1) % args.grad_accum == 0 or batch_i + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            if (batch_i + 1) % 250 == 0:
                print(
                    f"epoch={epoch + 1} batch={batch_i + 1}/{len(batches)} "
                    f"crop={max_residues} loss={running / 250:.4f} "
                    f"elapsed={(time.time() - started) / 60:.1f}m", flush=True
                )
                running = 0.0

        score = evaluate(model, tokenizer, sequences, targets, val_indices, 16)
        print(f"epoch={epoch + 1} full-length val rho={score:.6f}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            model.save_pretrained(args.output, safe_serialization=True)
            tokenizer.save_pretrained(args.output)
            print(f"saved new best to {args.output}", flush=True)
    print(f"best rho={best_score:.6f} epoch={best_epoch + 1}")


if __name__ == "__main__":
    main()
