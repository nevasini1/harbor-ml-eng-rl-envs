# Harbor ML-engineering RL environments

Four agent-evaluation environments for ML engineering: give an agent a pretrained model, a
fixed compute budget and labelled data, and score whether it can *adapt* the model — not
whether it can call `fit()`.

The hard part is not the task. It is the **reward**: making a number that goes up only when
the agent does the work you meant to test, and that says the same thing twice in a row.
Most of this repo is the evidence behind the anchors that number is built from.

```
tasks/      the four tasks, one directory each        -> tasks/README.md
common/     the verifier core all of them share
research/   how the splits were cut, how the anchors were measured
jobs/       Harbor run outputs — the record of what was actually run
docs/       background reading
```

---

## The tasks

| | tests | compute | separation | oracle |
|---|---|---|---|---|
| [`mol-property-adapt`](tasks/mol-property-adapt/) | encoder adaptation on molecules | 8 CPU, 4 h | 6.5σ / 4.1σ | 0.9097 |
| [`qa-sft-adapt`](tasks/qa-sft-adapt/) | supervised fine-tuning of a 135M causal LM | 8 CPU, 4 h | 6.0σ / 16.0σ / 5.6σ | **1.0** |
| [`pref-reward-model`](tasks/pref-reward-model/) | reward modelling on human preferences | 1 GPU, 4 h | 3.1σ — **fails the bar** | 0.5828 |
| [`sciml-protein-regression`](tasks/sciml-protein-regression/) | **shelved** — proteins; the negative result | 1 GPU, 4 h | inverted | 1.0, and so does a frozen probe |

All four have been run end to end through Harbor with their own shipped oracle — agent
container, artifact hand-off, verifier, reward. **The three oracle rewards differ for
reasons worth understanding rather than fixing:**

- **1.0** on `qa-sft-adapt` — the oracle cleared its reference on every eval set.
- **0.58** on `pref-reward-model` — it landed 1.3σ under a reference that is a five-seed
  *mean*, which a single seed does about half the time.
- **1.0** on the protein task means nothing at all, because a submission that never touches
  the encoder scores 1.0 there too.

Each task's README carries its measured ladder, its integrity layers and its honest limits.

---

## The reward

Every active task scores the same way. Per eval set, the raw metric is normalized onto
[0,1] between two measured anchors:

```
recovery = clip((metric − base) / (reference − base), 0, 1)
reward   = integrity_gate × mean(recovery over eval sets)
```

- **`base`** earns **0**: the *ceiling* of everything that does **not** adapt the model —
  the best of the trivial methods, including ones that are not models at all.
- **`reference`** earns **1**: a tuned, deliberately ordinary adaptation.
- **`integrity_gate`** is 0 if provenance, contamination or shape checks fail, so a zero
  always carries an attributable reason instead of being indistinguishable from a weak model.

Both anchors are **measurements** over 5 seeds on the private split, each re-derivable from
a committed script. The uncapped `recovery_raw` is recorded beside the capped value,
because the clip is what hides a mis-set anchor.

### Whether an eval set may ship

One rule, in one place — [`common/shipping.py`](common/shipping.py) — applied to every task,
including ones that shipped before it existed. It starts from a stated tolerance rather
than a chosen threshold:

```
MAX_REWARD_NOISE = 0.25      ->      band_sigma >= 4.0
```

"Rerunning the same submission with a different seed must not move its reward by more than
a quarter." Since recovery is `(score − base) / band`, that tolerance *is* the bar. Change
the tolerance and the bar moves with it, visibly.

Three further tests: the band must be positive; it must be significantly non-zero after a
Bonferroni correction for how many eval sets were screened (shipping the best of k inflates
the winner); and the reference must beat a randomly-initialised control by more than one
sigma. `python common/shipping.py` prints the verdict for every task.

| task | eval set | band σ | reward noise on a rerun | verdict |
|---|---|---|---|---|
| mol-property-adapt | `tox21` | 6.48σ | 0.15 | ships |
| mol-property-adapt | `bbbp` | 4.09σ | 0.24 | ships |
| qa-sft-adapt | `sciq` | 15.98σ | 0.06 | ships |
| qa-sft-adapt | `arc_easy` | 6.02σ | 0.17 | ships |
| qa-sft-adapt | `openbookqa` | 5.63σ | 0.18 | ships |
| pref-reward-model | `helpful_rs` | 3.10σ | 0.32 | **fails: imprecise** |

![which eval sets may be used as a reward](research/results/shipping_criterion.png)

Left panel: the base→reference band *is* the scoring range, so a submission's reward is its
position along it. Right panel: the same nine eval sets against the bar. The preference
track's bands are visibly hair-thin next to the SFT track's — and its `helpful_rs` band is
statistically real (z = 5.5) yet still unusable, which is the distinction the two tests
exist to draw. Regenerate with `research/.venv/bin/python research/plot_criterion.py`;
it reads the committed anchor files rather than hardcoded numbers.

The earlier bar was `band_sigma >= 3.0`, picked by going one notch below what the repo had
already shipped, and a second criterion was **removed after it excluded the eval set I
wanted to keep**. `pref-reward-model` passed under that and does not pass now. That is the
point of deriving the bar instead of choosing it.

---

## What is shared

The mol task's grader was the only one whose behaviour had been measured end to end through
Harbor, so it became the library rather than being reimplemented three times.

| | |
|---|---|
| [`common/verifier_core.py`](common/verifier_core.py) | integrity layers, fail-closed anchor loader, recovery normalization, and the driver that guarantees a reward is always written |
| [`common/textmatch.py`](common/textmatch.py) | shingle-overlap contamination — the text analogue of the mol task's InChIKey check |
| [`common/regrade.sh`](common/regrade.sh) | re-score a finished trial against a rebuilt verifier image |
| [`common/sync.py`](common/sync.py) | copies the shared modules into each build context; `--check` fails on drift |

The extraction is checked, not asserted: regrading `jobs/mol-oracle-modal` through the
refactored grader returns **0.909654** with byte-identical per-eval-set metrics. (The
recorded 0.909661 came from x86 Modal; the 7×10⁻⁶ difference is arm64 float, and the
*original* grader reproduces it too.)

**Four integrity layers, failing on disjoint inputs:** architecture-config hash; sha256 vs
pinned public checkpoints, comparing the repo for *equality* not prefix; per-tensor float64
cosine vs the base body; and nearest-ancestor, which is the only one that catches a
laundered instruct checkpoint. The cosine layer deliberately **allows** an unmodified body
— freezing the backbone is legitimate, and the anchors are what make it score zero.

Verifier suites run each real image under `--network none` against constructed fixtures —
honest base, shuffled tensor, NaN'd tensor, deleted config, bit-identical public twin,
laundered sibling, contaminated log. **The accept path is asserted first**, because a false
reject is indistinguishable in the reward from an agent that did nothing.

---

## What this repo learned about reward design

Ten eval sets across four tracks failed in different ways, and the failures rhyme.

1. **Measure the whole ladder, not two anchors.** random-init → trivial heuristic → frozen
   probe → frozen + trained head → full adaptation. Check the ordering is monotone and that
   adjacent rungs are separated by ≫ noise.
2. **`base` is the ceiling of the trivial class — and that class contains things that are
   not models.** On the preference track the ceiling turned out to be a heuristic with no
   parameters: "pick the longer response" scored **0.6031** against a full fine-tune's
   0.6042. The protein task pinned its lower tier to a ridge probe at 0.3887 while a trained
   head reached 0.546, above its *top* tier.
3. **Always run the random-init arm, and require its gap to clear noise.** An eval set can
   show a clean band while the pretrained weights contribute nothing, because both arms
   learn the same surface feature. A randomly initialized encoder scored 0.5938 against the
   pretrained fine-tune's 0.6042 — a gain smaller than seed noise.
4. **The size of the held-out set is a reward parameter.** It sets the noise floor of every
   score the environment will ever produce. Seed spread at n=1,000 was 0.015 against a
   binomial floor of 0.0156 — the noise was *measurement*, not training. A 4× larger
   holdout is what made `helpful_rs` shippable.
5. **Measure the verifier the way it actually runs.** A micro-benchmark on the first 128
   rows predicted 3.7 min per eval set; the real run took **1,933 s for two**. Verifier
   wall time is what bounds the holdout, so being wrong about it by 4× is a design error.
6. **An anchor that is a mean is not a bar a single run clears.** `reference` is a five-seed
   mean, so the shipped oracle lands below it about half the time by construction. Read an
   oracle trial as "near 1.0, often under".
7. **A task's compute class is not one setting.** Moving a task to a GPU means `gpus = 1`,
   *and* the torch wheel index in the environment image, *and* a `.to(device)` in the
   reference script. Missing either pins the reference to CPU — 6.7 h against a 4 h timeout,
   failing as a silent wall-clock timeout. Only an end-to-end run finds it.
8. **Anchors are functions of the split and the training size.** Re-measure after changing
   either. Every failure here appeared *after* screening, once both were locked.
9. **Removing leakage ≠ increasing difficulty.** Both scientific tracks found the principled
   split made the task *easier*.
10. **Never clip silently.** Log the uncapped recovery. The clip is what hid a
    mis-calibrated reference for an entire working session.
11. **Fail closed on your own configuration.** Missing anchors, held-out keys or sibling
    checkpoints are a broken image, not a default. Robustness to *agent* input and to *your
    own* config are different problems.
12. **Separate measuring from deciding.** `modal_measure.py` records what every arm scored;
    `finalize_anchors.py` turns that into anchors by stated rules. That is the difference
    between an anchor that is measured and one chosen once in a session nobody kept.

---

## Running things

The task trees need their fixtures fetched first — pinned checkpoints too large for git:

```bash
python research/posttrain/assemble_tasks.py   # anchors, private rows, base + siblings
python common/sync.py --check                 # shared grader modules have not drifted
```

Then:

```bash
# oracle trial through Harbor on Modal. Local Docker on macOS silently ignores
# network_mode = "no-network", so the verifier would not be isolated.
harbor run -c tasks/<task>/configs/job-modal.json --agent oracle

# re-score a finished trial without re-running the agent
./tasks/<task>/scripts/regrade.sh --all

# verifier regression suites, against the real images
python research/posttrain/verify_graders.py
modal run tasks/sciml-protein-regression/scripts/modal_verify_hardening.py
```

Re-deriving the post-training anchors from scratch is documented in
[`research/posttrain/RESULTS.md`](research/posttrain/RESULTS.md); the mol and protein
measurements are in [`research/SPIKE_RESULTS.md`](research/SPIKE_RESULTS.md).

---

## Open questions

1. **tox21's oracle does not reproduce its own anchor** — 0.6896 against
   `reference_auc = 0.7019`, below the minimum of the 5 seeds that set it. The shipped
   script and the anchor arm are the same recipe in two implementations, and the repo
   currently asserts both.
2. **`pref-reward-model` ships on one eval set at 3.10σ, chosen as the best of four.** A
   third of that reward band is seed noise, and taking the maximum of four marginal
   measurements inflates the figure — read it as "around 3σ". It is the weakest thing here.
3. **`base` is a max over noisy means, which biases it upward.** On `arc_easy` the two
   no-adaptation arms are 0.0013 apart with σ 0.0154, so which one wins is near a coin flip.
   The bias is in the safe direction; a proper treatment would use an upper confidence bound.
4. **The protein task's GPU oracle beat its frozen probe** — 0.5733 against 0.5358, the
   opposite of the multi-seed result the shelving verdict rests on. One seed is not enough
   to reopen it, but the ordering should be re-measured multi-seed on GPU.
5. **The agent gets one shot at a final artifact.** RE-Bench scores the best entry in an
   intermediate score log; that is not portable here without exposing the held-out set. So
   these tasks partly measure "did you select on validation" alongside "can you post-train".
6. **The shingle fingerprint is subsampled 1-in-4** to keep it under 4 MB in the verifier
   image. A 30-token leak is caught with probability 99.6%; a one-sentence leak is not.
7. **`MIN_ENCODER_TENSORS = 50` is still hardcoded in the mol grader.** The newer tasks take
   90% of the base's own body-tensor count, which survives a model swap.

---

## Note on private holdout

This public repo includes the private seeds and the held-out rows by request. That publishes
the graded holdout; **do not treat the reward as un-gameable if agents can read this
repository.** The in-container defences — no verifier network, contamination detection, and
the implausible-score tripwire — assume the agent cannot see it.

Paths inside `jobs/` point at the pre-reorganisation layout on purpose: those files are
evidence of runs that happened, not live configuration.
