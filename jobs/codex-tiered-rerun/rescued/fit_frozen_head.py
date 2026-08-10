import argparse
import copy
import json
import random

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class ExactEsmHead(torch.nn.Module):
    """Dropout is zero in the supplied ESM config, leaving dense/tanh/out_proj."""

    def __init__(self, width=320):
        super().__init__()
        self.dense = torch.nn.Linear(width, width)
        self.out_proj = torch.nn.Linear(width, 1)

    def forward(self, x):
        return self.out_proj(torch.tanh(self.dense(x))).squeeze(-1)


def rho(y, pred):
    return float(spearmanr(y, pred).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="/app/embeddings.npz")
    ap.add_argument("--output", default="/app/frozen_head.pt")
    args = ap.parse_args()
    torch.set_num_threads(8)
    data = np.load(args.embeddings, allow_pickle=True)
    y = data["target"].astype(np.float32)
    train = data["split"] == "train"
    val = ~train

    diagnostics = {}
    ridge_head_candidates = []
    for pool_name in ("cls", "mean"):
        values = data[pool_name]
        for layer in range(values.shape[1]):
            scaler = StandardScaler()
            xt = scaler.fit_transform(values[train, layer])
            xv = scaler.transform(values[val, layer])
            for alpha in (100.0, 1000.0, 10000.0):
                model = Ridge(alpha=alpha).fit(xt, y[train])
                diagnostics[f"{pool_name}_L{layer}_a{alpha:g}"] = rho(
                    y[val], model.predict(xv)
                )
                if pool_name == "cls" and layer == values.shape[1] - 1:
                    # Fold StandardScaler into a raw-embedding linear model.
                    raw_weight = model.coef_ / scaler.scale_
                    raw_bias = float(model.intercept_ - np.dot(raw_weight, scaler.mean_))
                    ridge_head_candidates.append((
                        diagnostics[f"{pool_name}_L{layer}_a{alpha:g}"],
                        raw_weight.astype(np.float32), raw_bias, alpha,
                    ))
    for alpha in (100.0, 1000.0, 10000.0):
        values = np.concatenate((data["mean"][:, -1], data["last_std"]), axis=1)
        scaler = StandardScaler()
        xt = scaler.fit_transform(values[train])
        xv = scaler.transform(values[val])
        model = Ridge(alpha=alpha).fit(xt, y[train])
        diagnostics[f"mean_std_a{alpha:g}"] = rho(y[val], model.predict(xv))
    print(json.dumps(dict(sorted(diagnostics.items(), key=lambda x: -x[1])), indent=2))

    # Optimize the precise deployable head on the unmodified final CLS embedding.
    x_train = torch.from_numpy(data["cls"][train, -1].copy())
    x_val = torch.from_numpy(data["cls"][val, -1].copy())
    y_mean = float(y[train].mean())
    y_std = float(y[train].std())
    y_train = torch.from_numpy(((y[train] - y_mean) / y_std).copy())
    # Implement the best final-CLS Ridge with tanh kept in its linear region.
    ridge_head_candidates.sort(key=lambda item: -item[0])
    ridge_score, ridge_weight, ridge_bias, ridge_alpha = ridge_head_candidates[0]
    epsilon = 0.01
    ridge_state = {
        "dense.weight": torch.eye(320) * epsilon,
        "dense.bias": torch.zeros(320),
        "out_proj.weight": torch.from_numpy(
            (ridge_weight / (epsilon * y_std))[None, :].copy()
        ),
        "out_proj.bias": torch.tensor([(ridge_bias - y_mean) / y_std]),
    }
    ridge_deploy = ExactEsmHead()
    ridge_deploy.load_state_dict(ridge_state)
    with torch.no_grad():
        exact_ridge_score = rho(y[val], ridge_deploy(x_val).numpy())
    print(f"deployable ridge alpha={ridge_alpha:g} rho={exact_ridge_score:.6f}")
    best_global = (exact_ridge_score, ridge_state, -1, 0)
    for seed in (17, 29, 43):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        head = ExactEsmHead()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
        gen = torch.Generator().manual_seed(seed)
        best = (-1.0, None, 0)
        stale = 0
        for epoch in range(250):
            order = torch.randperm(len(x_train), generator=gen)
            head.train()
            for start in range(0, len(order), 256):
                inds = order[start : start + 256]
                pred = head(x_train[inds])
                loss = torch.nn.functional.smooth_l1_loss(pred, y_train[inds], beta=0.5)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if epoch % 2 == 0:
                head.eval()
                with torch.no_grad():
                    pred = head(x_val).numpy()
                score = rho(y[val], pred)
                if score > best[0]:
                    best = (score, copy.deepcopy(head.state_dict()), epoch)
                    stale = 0
                else:
                    stale += 1
                if stale >= 20:
                    break
        print(f"seed={seed} best_rho={best[0]:.6f} epoch={best[2]}")
        if best[0] > best_global[0]:
            best_global = (best[0], best[1], seed, best[2])
    torch.save(
        {"state_dict": best_global[1], "y_mean": y_mean, "y_std": y_std,
         "val_rho": best_global[0], "seed": best_global[2], "epoch": best_global[3]},
        args.output,
    )
    print(f"saved {args.output}; best={best_global[0]:.6f}")


if __name__ == "__main__":
    main()
