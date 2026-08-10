# `sciml-protein-regression` — shelved

Harbor task (1 GPU, 4 h): fine-tune `facebook/esm2_t6_8M_UR50D` on a Meltome
thermostability split; score Spearman on a private held-out set.

**Status: shelved. Do not use this reward.** It is kept because its verifier is the
better-hardened of the originals and because the negative result is worth reading — it is
the reason every other task in this repo scores on a continuous recovery between measured
anchors instead of on fixed tiers.

---

## Why it is shelved

### 1. The ordering is inverted

Across two splits, 13 frozen seeds and 7 fine-tune seeds at std ~0.003:

| split | frozen probe | fine-tune | gap |
|---|---|---|---|
| shipped | 0.5494 ± 0.0033 | 0.5214 ± 0.0017 | −0.028 (−7.5σ) |
| MMseqs2 cluster, 30% id | 0.5784 ± 0.0029 | 0.5257 ± 0.0032 | −0.053 (−12.2σ) |

Fine-tuning reliably *loses*, so no threshold placement repairs it. Two independent
confirmations: [Kumar et al. 2022](https://arxiv.org/abs/2202.10054) (fine-tuning distorts
pretrained features under distribution shift) and
[iScience 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/), which finds head-only
*wins* on melting-point prediction specifically.

The cluster-split row is the counter-intuitive one: MMseqs2 cut prefix leakage from 89
sequences to 3 and the frozen probe got **better**. Assigning whole clusters to test groups
related proteins with similar Tm together, which is easier to rank. **Removing
near-duplicates and making a split harder are not the same thing.**

### 2. The reward is the wrong shape

Three fixed tiers rather than a recovery between measured anchors:

```
reward = 0.0  if integrity fails, or spearman < t_weak     (0.3887)
       = 0.5  if t_weak <= spearman < t_strong             (0.45)
       = 1.0  if spearman >= t_strong
```

Both thresholds are mis-set and the repo proves it. `scripts/probe_ceiling.json` records
`tiers_json_claims_frozen_probe: 0.3887` beside a mean-pool Ridge at **0.4586**, a
nonlinear head at **0.4973** and a trained head at **0.546** — three frozen methods, none
of which adapts the encoder, two of which clear `t_strong = 0.45` outright.

`t_weak` was calibrated; `t_strong` was **chosen** — its own note says "a fixed
strong-oracle bar below the successful Codex run". No script produces it.

Quantization makes every calibration error maximal: a submission 0.001 past a threshold
scores identically to one 0.1 past, and a run near a boundary flips tiers between reruns
for a reward swing of 0.5. **Tiers look robust because they hide small errors, and are
fragile for exactly that reason.**

---

## What was actually run

Both known artifacts, re-scored through the hardened grader on Linux:

| artifact | Spearman | reward | cosine_min | tensors |
|---|---|---|---|---|
| tier-0.5 lock | 0.43121570 | 0.5 | 0.99979 | 108 |
| **frozen probe** (no encoder adaptation) | 0.53577846 | **1.0** | 1.00000 | 108 |

Both reproduce their recorded values to the 7th decimal. The second row is the problem,
not a pass: a submission that never touches the encoder takes the maximum reward.

The oracle has since been run on GPU (`jobs/protein-oracle-gpu/`) — which is what the task
was given a GPU for, since its 4-epoch fine-tune could not finish inside its own budget on
CPU. It scored **Spearman 0.5733, reward 1.0**, at `cosine_min` 0.9838 over 108 tensors.

That is the one number that cuts **against** the shelving argument: 0.5733 for a fine-tune
is above the 0.5358 the frozen-probe artifact scored on the same split. It does not
overturn the verdict — the inversion was measured over 13 frozen and 7 fine-tune seeds and
this is one seed of a different recipe — but the ordering deserves a multi-seed
re-measurement on GPU before the task is written off for good. It changes nothing about
the reward shape, which is the other and independent reason to shelve it.

---

## Layout

```
task.toml
instruction.md
environment/          # agent image (base model + /data/train.csv.gz)
solution/solve.sh     # oracle (4-epoch full fine-tune)
tests/                # separate verifier image (private_test + grade.py)
scripts/              # split construction, calibration, Modal measurement probes
```

## Run

```bash
harbor run -c tasks/sciml-protein-regression/configs/job-modal-oracle.json --agent oracle
./tasks/sciml-protein-regression/scripts/regrade.sh --all

# 16 verifier hardening assertions against the real image
modal run tasks/sciml-protein-regression/scripts/modal_verify_hardening.py
```

Regenerate the private split with `python scripts/make_private_split.py`; the seed lives
only in that script and never enters the agent image.

## Notes

- Hydro was measured in [`research/`](../../research/SPIKE_RESULTS.md) and failed the
  headroom gate (combinatorial library; one-hot saturates, WT-transfer fine-tune does not
  climb). This task uses Meltome-mixed with a fresh unpublished resplit.
- Harbor's local Docker provider silently ignores `network_mode = "no-network"` on macOS,
  so the verifier is only genuinely isolated on Linux or a provider with real egress
  control. Use Modal/E2B.
- `reward.json` values must all be numeric (Harbor `VerifierResult`); string fields go to
  `reward_meta.json`.
