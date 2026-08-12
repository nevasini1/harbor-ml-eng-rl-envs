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
docs/       background reading, and the reward's decision log
              -> docs/decisions/  (13 records, superseded ones kept and marked)
```

Every claim below about how the reward is computed has a record behind it in
[`docs/decisions/`](docs/decisions/) saying when it was decided, what it replaced, and
what moved as a result. Seven of the thirteen moved a number that had already shipped;
[the index](docs/decisions/) lists which, and the reader impact of each.

---

## The tasks

| | tests | compute | separation | oracle | **agent** |
|---|---|---|---|---|---|
| [`mol-property-adapt`](tasks/mol-property-adapt/) | encoder adaptation on molecules | 8 CPU, 4 h | 7.6σ / 6.8σ | 0.9097 | **1.0** |
| [`qa-sft-adapt`](tasks/qa-sft-adapt/) | supervised fine-tuning of a 135M causal LM | 8 CPU, 4 h | 6.0σ / 16.0σ / 5.6σ | 1.0 | **0.734** |
| [`pref-reward-model`](tasks/pref-reward-model/) | reward modelling on human preferences | 1 GPU, 4 h | 3.1σ — **fails the bar** | 0.5828 | **0.865** † |
| [`sciml-protein-regression`](tasks/sciml-protein-regression/) | **repairable** — proteins; the task discriminates, the reward does not | 1 GPU, 4 h | 4.8σ (re-measured) | 1.0 | 1.0 — and so does a frozen probe |

All four have been run end to end through Harbor, by their own shipped oracle **and** by a
real agent (`codex`, gpt-5.6-sol). † `pref-reward-model` was graded 0.0 at the time by a
false reject in the verifier and is 0.865 after the fix — see
[What the agent trials found](#what-the-agent-trials-found).

**The oracle rewards differ for reasons worth understanding rather than fixing:**

- **1.0** on `qa-sft-adapt` — the oracle cleared its reference on every eval set.
- **0.58** on `pref-reward-model` — it landed 1.3σ under a reference that is a five-seed
  *mean*, which a single seed does about half the time.
- **1.0** on the protein task means nothing at all, because a submission that never touches
  the encoder scores 1.0 there too — its three fixed tiers both sit below the frozen ceiling,
  so a real 0.0375 difference between the oracle and a frozen probe quantizes away.

The agent rewards are the more useful signal, because an oracle never reads
`instruction.md`. `qa-sft-adapt` is the one that behaved best as an *evaluation*: it
neither clipped to 1.0 nor floored to 0, landing at 0.734 with the shortfall
concentrated on its hardest eval set.

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
  It is not a variable: it is the `Reject` path in [`common/verifier_core.py`](common/verifier_core.py),
  which floors an eval set to 0 and records `status` and `reason` in `metrics.json`. The gate is
  **per submission, not per eval set** — `mol-property-adapt` accepts a separate model per eval
  set and checks all of them before scoring any, so substituting one earns nothing on the
  others. It used to be checked per eval set there, which paid mean(0, 1) = 0.5 for a
  half-substituted submission. Note a 0 does not only mean rejection: a model that raises at
  inference also floors to 0, recorded as `status: "error"` rather than `"rejected"`.

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
| mol-property-adapt | `tox21` | 7.62σ | 0.13 | ships |
| mol-property-adapt | `bbbp` | 6.81σ | 0.15 | ships |
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

**σ used to mean two things.** The mol figures above were 6.48σ and 4.09σ until
`research/assemble_task.py` was put on `common/shipping.py`. They divided the band by the
two arms' noise **added in quadrature**, where the shared rule takes **the larger of them** —
and on `bbbp` the quadrature used the frozen *head*'s noise even though the shipped `base` is
the deterministic logistic probe. Nothing shipped or failed differently, because quadrature
is the more conservative of the two and both cleared 4.0 either way; what was broken was
comparison, and these numbers are compared against each other constantly. The band, the
anchors and every reward are unchanged. `common/check_reward.py` now fails if the two ever
diverge again.

The protein task's re-measured band carries the same correction: `scripts/lpft.json` records
3.92σ by quadrature, which is **4.84σ** under the shared rule — so that band clears the bar
on its own. It is not shipped as a reward, because that task scores by three fixed tiers.

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

**Four integrity layers, failing on disjoint inputs** — but not four on every task, which the
plain sentence used to imply:

| | mol | pref | qa | protein |
|---|---|---|---|---|
| architecture-config hash | ✓ | ✓ | ✓ | ✓ |
| per-tensor float64 cosine vs the base body | ✓ | ✓ | ✓ | ✓ |
| sha256 vs pinned public checkpoints | ✓ | ✓ | ✓ | 2 hardcoded hashes |
| **nearest-ancestor** | — | — | ✓ | — |
| contamination scan | ✓ | ✓ | ✓ | score tripwire only |

Nearest-ancestor is the only layer that catches a laundered instruct checkpoint, and it runs on
one task, because `qa-sft-adapt` is the only one with a `siblings/` fixture. That is a fixture
gap, not a design decision — `distilroberta-base` in particular has many public fine-tunes to
launder. The protein grader has no shingle or fingerprint scan at all; its only contamination
defence is the implausible-score tripwire. The cosine layer deliberately **allows** an unmodified body
— freezing the backbone is legitimate, and the anchors are what make it score zero.

Verifier suites run each real image under `--network none` against constructed fixtures —
honest base, shuffled tensor, NaN'd tensor, deleted config, bit-identical public twin,
laundered sibling, contaminated log. **The accept path is asserted first**, because a false
reject is indistinguishable in the reward from an agent that did nothing.

---

## What the agent trials found

Every result above the agent column is from `codex` (gpt-5.6-sol) on Modal, one trial each,
reading only `instruction.md`. Until these ran, the three active tasks had **only ever seen
`--agent oracle`** — which executes `solution/solve.sh` and therefore tests the plumbing,
not the task. Three trials changed more than fifty measurements had.

| task | reward | runtime | eval-set detail |
|---|---|---|---|
| `mol-property-adapt` | **1.0** | 1h 22m | tox21 0.7209 (raw recovery **1.280**), bbbp 0.9188 (raw **1.472**) |
| `qa-sft-adapt` | **0.734** | 2h 05m | arc_easy 0.683 → 0.867, sciq 0.815 → 0.908, openbookqa 0.337 → **0.427** |
| `pref-reward-model` | 0.0 → **0.865** | 49m | helpful_rs 0.6242 → 0.8645, after the fix below |

### A false reject, and it was intermittent

`pref-reward-model`'s agent read the contract, built a prompt-level validation split,
fine-tuned the provided base — and was scored **0.0**:

```
encoder lineage check failed: min per-tensor cosine 0.8541 < 0.9
  on encoder.layer.0.attention.self.key.bias
```

Measured over that submission: **51 weight matrices at cosine ≥ 0.9999** (median 1.0000),
and exactly **1 of 100** tensors under the floor — a 768-element attention key bias whose
entries are near zero, so a functionally irrelevant update rotates it a long way. Key
biases are inert enough that several implementations omit them entirely.

The floor now applies to **weight matrices only**; 1-D cosines are reported as
`min_vector_cosine` and no longer reject. Verified afterwards: the mol oracle still
regrades to 0.909654 with identical metrics, and `shuffled`, `nan`, `truncated`,
`public_twin` and `contaminated` are all still rejected.

Two things make this worth recording rather than just fixing:

- **It was intermittent.** The mol agent's 1-D minimum was 0.9998 and it sailed through; the
  reward-model agent trained a fresh scalar head through a ranking loss on a GPU and moved
  that bias far. A verifier that zeroes *some* honest submissions, on a vector that does not
  affect the model's output, corrupts the score distribution quietly instead of failing loudly.
- **My own fixtures could not have caught it.** I tested the two extremes — an untouched base
  at cosine 1.0 and a shuffled embedding at 0.007 — and never an honest fine-tune trained
  harder than my own oracle. This is precisely the failure the grader's docstring warns
  about: a false reject is indistinguishable, in the reward, from an agent that did nothing.

### `qa-sft-adapt` is the one that behaved like an evaluation

It neither clipped to 1.0 nor floored to 0. It landed at 0.734 with the shortfall
concentrated on `openbookqa` — the hardest set, where the base is barely above chance — and
recovered 0.87 and 0.91 on the other two. That is discrimination, which is what an eval set
is for.

Its agent also **out-designed my oracle**, unprompted. My `train_reference.py` holds back
398 items at random, and I showed earlier that its epoch-2-vs-3 choice was a 0.23σ coin
flip. The agent chose **598 items, source-stratified**, and ran a real ablation: it measured
the untouched base, compared source-balanced against natural-frequency sampling, then tested
and *rejected* three things on evidence — distractor-ranking (a 0.0008 tie), MMLU science
pre-training (hurt validation), and OpenBookQA-only continuation (overfit). Excluding three
ideas with the measurement for each is better discipline than the shipped oracle's.

It also came in **below** its own local validation on two of three sets (ARC 0.733 → 0.683,
OBQA 0.416 → 0.337), so the region-holdout split is genuinely harder than a local one — the
split design is doing work.

### What else the trials established

- **The submission contract is followable from the prose.** Three agents, three valid
  `save_pretrained` submissions, zero shape or contract failures.
- **No contamination false positives.** Overlap 0 everywhere. The mol InChIKey scan parsed
  one molecule out of real agent output and correctly did not flag it.
- **Rule 4 is followable but not reliable.** mol and QA wrote `train_log.txt`; the
  reward-model agent did not. When it is skipped, the contamination scan sees only the model
  directory — and `/logs/agent` is the surface it deliberately scans *first*.
- **bf16 matters.** The QA agent trained in bf16. Without the earlier switch of
  `load_tensors` to read safetensors through torch, that trial would have died with
  `TypeError: data type 'bfloat16' not understood` instead of scoring 0.734.
- **mol's reference now looks soft.** An agent beat a 25-seed-calibrated reference by 28% of
  the band on tox21 and 47% on bbbp, and its 0.7209 exceeds the best of the 25 seeds
  (0.7111) that set the anchor. A task where a first attempt clips the ceiling has lost its
  headroom; that anchor wants re-measuring.

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
12. **An oracle validates the plumbing; only an agent validates the task.** Three agent
    trials found a false reject that fifty measurements and a purpose-built adversarial
    fixture suite had missed — because I had only ever tested the extremes, never an honest
    fine-tune trained harder than my own reference. Run an agent before believing an
    environment works.
13. **Separate measuring from deciding.** `modal_measure.py` records what every arm scored;
    `finalize_anchors.py` turns that into anchors by stated rules. That is the difference
    between an anchor that is measured and one chosen once in a session nobody kept.
14. **A re-measurement needs a control arm that reproduces the number being re-measured,
    and the check has to be stated before the results are read.** Re-measuring mol's
    reference meant transcribing the shipped recipe into a new script — and the
    transcription was wrong in three places at once (batch size, validation fraction, loss
    normalisation). Every candidate was then compared against a baseline the task does not
    have, and **the ordering reversed**: the arm that looked like a clear win became the
    worst one once the control reproduced 0.7019. A re-measurement without a control does
    not measure the thing it names; it measures your transcription of it.

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

1. **tox21's oracle does not reproduce its own anchor**, and five explanations have been
   ruled out. The anchor itself is sound — re-running `modal_legal_anchors.py` unmodified
   returns 0.7024 against the committed 0.7019, seeds 0/2/3/4 exact. But the shipped
   `train_reference.py` (0.6897), an independent reimplementation (0.6896 ± 0.0037 over
   5 seeds) and the oracle's own saved checkpoint (0.689651) all agree with each other and
   not with it, with **no overlap** across 5 seeds each. Eliminated by measurement: loss
   normalisation, a moved training split, the AUC task-skip rule, eval batch size, and the
   logits-vs-sigmoid score transform. Every *evaluation-side* candidate is gone, so despite
   the non-overlapping spread the cause is training-side; next step is a component bisect,
   not another hypothesis.
2. **`pref-reward-model` ships on one eval set at 3.10σ, chosen as the best of four.** A
   third of that reward band is seed noise, and taking the maximum of four marginal
   measurements inflates the figure — read it as "around 3σ". It is the weakest thing here.
3. **mol's reference is too soft, and could not be raised on evidence.** An agent's first
   attempt beat it by 28% of the band on tox21 and 47% on bbbp; both eval sets clipped at
   recovery 1.0, so the task no longer discriminates at the top. Two stronger candidate
   recipes were measured, 5 seeds each — held-out *scaffold-group* validation, and the
   agent's own LRs/schedule/class weights — and **neither beat the shipped recipe on either
   eval set**; grouped validation was worse on both. So the defect is real but the anchor is
   not the fix, and raising it would invent a number no script emits. A first pass at this
   was **void** — its control arm did not reproduce the shipped number, and under it the
   losing arm looked like a clear win. See
   [`mol_reference_candidates.json`](research/results/mol_reference_candidates.json).
4. **`base` is a max over noisy means, which biases it upward.** On `arc_easy` the two
   no-adaptation arms are 0.0013 apart with σ 0.0154, so which one wins is near a coin flip.
   The bias is in the safe direction; a proper treatment would use an upper confidence bound.
5. **The protein task's GPU oracle beat its frozen probe** — 0.5733 against 0.5358, the
   opposite of the multi-seed result the shelving verdict rests on. One seed is not enough
   to reopen it, but the ordering should be re-measured multi-seed on GPU.
6. **The agent gets one shot at a final artifact.** RE-Bench scores the best entry in an
   intermediate score log; that is not portable here without exposing the held-out set. So
   these tasks partly measure "did you select on validation" alongside "can you post-train" —
   and the QA agent's run is evidence it matters: it chose a 598-item stratified split where
   the shipped oracle uses 398 at random.
7. **The shingle fingerprint is subsampled 1-in-4** to keep it under 4 MB in the verifier
   image. A 30-token leak is caught with probability 99.6%; a one-sentence leak is not.
8. **`MIN_ENCODER_TENSORS = 50` is still hardcoded in the mol grader.** The newer tasks take
   90% of the base's own body-tensor count, which survives a model swap.

---

## Note on private holdout

This public repo includes the private seeds and the held-out rows by request. That publishes
the graded holdout; **do not treat the reward as un-gameable if agents can read this
repository.** The in-container defences — no verifier network, contamination detection, and
the implausible-score tripwire — assume the agent cannot see it.

Paths inside `jobs/` point at the pre-reorganisation layout on purpose: those files are
evidence of runs that happened, not live configuration.
