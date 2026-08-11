# `sciml-protein-regression` — repairable

Harbor task (1 GPU, 4 h): fine-tune `facebook/esm2_t6_8M_UR50D` on a Meltome
thermostability split; score Spearman on a private held-out set.

**Status: the task discriminates; the reward does not.** This was previously written up as
"shelved — the ordering is inverted, fine-tuning reliably loses." That verdict was wrong,
for two independent reasons, both found by measurement. What is actually broken is the
three-tier reward, and that is fixable.

---

## The verdict that was wrong, and why

Two measurements shelved this task: `modal_variance.py` (frozen 0.546 vs fine-tune 0.5169)
and `modal_cluster_split.py`, which reproduced the sign on a second split at −7.5σ and
−12.2σ. Both were right about what they measured and wrong about what it meant.

**1. The fine-tune arm was pathological, not representative.** Both arms unfroze the top two
encoder layers starting from a *randomly initialised head* — the exact configuration
[Kumar et al. 2022](https://arxiv.org/abs/2202.10054) identify as destructive, because early
gradients are dominated by head error and wreck good pretrained features before the head is
any good. Their prescribed remedy is **LP-FT**: fit the head on frozen features first, then
warm-start a full fine-tune from it.

`solution/solve.sh` already did that, and nothing had ever been compared against it. This
README previously cited Kumar et al. as *confirming* the inversion. That was backwards — the
paper explains why the comparison arm was broken and prescribes the fix.

**2. The frozen baseline was off-contract.** `variance.json`'s frozen arm used mean pooling,
but `EsmClassificationHead.forward` is `x = features[:, 0, :]` — the CLS token. Mean pooling
is **not expressible in a legal submission**, so 0.546 was a ceiling drawn over methods no
agent could actually submit. The legal CLS frozen head is **0.5332**, and it cross-checks:
the real graded frozen artifact scored 0.5358, inside the CLS arm's range of 0.5294–0.5406,
while 0.546 sits outside it.

That is the same bug that made the mol anchors unreachable, in a second task, found the same
way — an anchor measured over a method the submission contract does not permit.

## Re-measured: both arms, same seeds, same protocol

From [`scripts/lpft.json`](scripts/lpft.json):

| arm | Spearman | n | |
|---|---|---|---|
| frozen head (CLS) | **0.5332 ± 0.0044** | 5 | ← `base` |
| **LP-FT** | **0.5627 ± 0.0061** | 5 | ← `reference` |
| naive top-2-layer fine-tune | 0.5169 ± 0.0053 | 4 | for contrast — the arm that produced the "inversion" |
| frozen head, mean-pooled (off-contract) | 0.546 ± 0.0054 | 8 | the old, unreachable baseline |

**band +0.0295, 3.92σ.** For scale, `mol-property-adapt`'s `bbbp` ships at 4.09σ. The
ordering is not inverted: adapting the encoder beats the frozen ceiling, when it is done the
way the literature says to do it.

## What is actually broken: the reward

Three fixed tiers on `tests/tiers.json` rather than a continuous recovery between measured
anchors:

```
reward = 0.0  if spearman < t_weak    (0.3887)
       = 0.5  if t_weak <= s < t_strong  (0.45)
       = 1.0  if spearman >= t_strong
```

**Both thresholds sit below the frozen ceiling of 0.5332.** So a submission that never
touches the encoder scores 1.0, which is exactly what the graded frozen-probe artifact does.

The GPU oracle scored **0.5733** and the frozen probe **0.5358** — a real 0.0375 difference,
0.85σ apart in a band the task genuinely resolves — and the tiers assign both **1.0**. The
signal exists and the reward quantizes it away.

## What it would take to ship

1. Continuous recovery in place of the three tiers, reusing `common/verifier_core.py` like
   the other three tasks.
2. `base = 0.5332`, `reference = 0.5627` from the measurement above.
3. A `train_reference.py` so the upper anchor is reproducible from a shipped script rather
   than chosen — the structure the mol task already has.
4. Re-approval by [`common/shipping.py`](../../common/shipping.py). At 3.92σ it currently
   fails the 4.0σ bar, narrowly.

## What was actually run

| artifact | Spearman | tiered reward | cosine_min | tensors |
|---|---|---|---|---|
| tier-0.5 lock | 0.4312 | 0.5 | 0.99979 | 108 |
| frozen probe (no encoder adaptation) | 0.5358 | **1.0** | 1.00000 | 108 |
| GPU oracle (LP-FT) | 0.5733 | **1.0** | 0.98378 | 108 |

Four `codex` agent trials also exist, in `jobs/2026-08-09__*` and
`jobs/codex-tiered-rerun/`. This is the only task in the repo that had seen a real agent
before this session, and one of those transcripts is why its hole was found: the agent
reasoned its way to a frozen-embedding baseline as the *safest* approach, and scored 1.0
without adapting the encoder.

## The lineage check, and a bug that was latent here

`tests/grade.py` used to carry its own ~90-line copy of the per-tensor cosine check, and that
copy had the bias false-reject fixed elsewhere in `8d42c88`: the floor applied to the minimum
over *all* shared tensors, including 1-D biases whose near-zero entries rotate under any
training. It now calls `common/verifier_core.check_lineage` instead.

How latent it was, measured on the two real submissions still on disk:

| submission | old min over all tensors | new min over weight matrices | which tensor drove the old min |
|---|---|---|---|
| GPU oracle (LP-FT) | 0.98378 | 0.99826 | a **1-D vector** |
| codex trial `BmGSL3y` | 0.99205 | 0.99945 | a **1-D vector** |

Both old values reproduce the `cosine_min` those trials actually recorded
(0.9837810800040203 and 0.9920478313094444), so the change is verified against real graded
artifacts rather than asserted. Verdicts unchanged, and the tensor count stays 108, so
`insufficient_backbone_overlap` is unaffected.

The point is what drove the number: this task's `cosine_min` has **always** been reported off
a 1-D vector, sitting ~0.09 above the floor while its weight matrices sat at 0.998+. An agent
that trained harder — the `pref-reward-model` one drove a bias to 0.854 — would have been
zeroed.

**Only the lineage check was migrated.** `check_architecture` returns a reason string rather
than raising, and `check_forbidden_hashes` compares against a hardcoded set of sibling
sha256s rather than the `public_hashes.json` index the shared layer expects — so it never had
the repo-prefix bug fixed in `8d42c88`. Migrating those means changing this grader's error
contract, which its named failure reasons are written against.

## Layout and running

```
task.toml          instruction.md        environment/
solution/solve.sh  LP-FT oracle          tests/       separate verifier image
scripts/           split construction, calibration, Modal measurement probes
```

```bash
harbor run -c tasks/sciml-protein-regression/configs/job-modal-oracle.json --agent oracle
./tasks/sciml-protein-regression/scripts/regrade.sh --all
modal run tasks/sciml-protein-regression/scripts/modal_verify_hardening.py   # NB: targets the MOL grader
modal run tasks/sciml-protein-regression/scripts/modal_lpft.py               # the re-measurement
```

Regenerate the private split with `python scripts/make_private_split.py`; the seed lives only
in that script and never enters the agent image.

Harbor's local Docker provider silently ignores `network_mode = "no-network"` on macOS, so
the verifier is only genuinely isolated on Linux or a provider with real egress control.
