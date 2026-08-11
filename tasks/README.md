# Tasks

Four Harbor tasks. Each is two containers — `environment/` becomes the agent's, `tests/`
becomes the verifier's — and only the paths listed in `task.toml`'s `artifacts` cross
between them.

| task | tests | compute | reward | separation | oracle | agent |
|---|---|---|---|---|---|---|
| [`mol-property-adapt`](mol-property-adapt/) | encoder adaptation on molecules | 8 CPU, 4 h | recovery between measured anchors | 7.6σ / 6.8σ | 0.9097 | **1.0** |
| [`qa-sft-adapt`](qa-sft-adapt/) | supervised fine-tuning of a 135M causal LM | 8 CPU, 4 h | same | 6.0σ / 16.0σ / 5.6σ | 1.0 | **0.734** |
| [`pref-reward-model`](pref-reward-model/) | **provisional** — reward modelling on human preferences | 1 GPU, 4 h | same | 3.1σ, fails the 4.0σ bar | 0.5828 | **0.865** |
| [`sciml-protein-regression`](sciml-protein-regression/) | **repairable** — encoder adaptation on proteins | 1 GPU, 4 h | 3 fixed tiers | 4.8σ (re-measured) | 1.0 | 1.0, and so does a frozen probe |

`separation` is `band_sigma`: the base→reference band divided by seed noise. Whether that
is good enough is decided by [`common/shipping.py`](../common/shipping.py), which derives
the bar (4.0σ) from a stated tolerance on reward noise (0.25) rather than choosing it.
`pref-reward-model` does not clear it and is assembled only under
`--allow-provisional`. `oracle` is the reward the
task's own shipped `solution/` earned through Harbor; `agent` is what `codex` (gpt-5.6-sol)
earned reading only `instruction.md` — see the root
[README](../README.md#what-has-actually-been-run) for why those three numbers differ and
why 1.0 is not always the good one.

## Layout of a task

```
<task>/
  README.md          what it measures, and what it cost to make it measure that
  task.toml          Harbor task definition: compute, timeouts, artifacts
  instruction.md     what the agent is told
  environment/       agent container: Dockerfile + training data + baked base model
  solution/          the oracle; the recipe that set the `reference` anchor
  tests/             verifier container: grader, private split, anchors, no network
  scripts/           regrade helper
  configs/           Harbor job configs
```

## Before building a verifier image

The three post-training and molecular tasks need their fixtures fetched — pinned
checkpoints too large for git:

```bash
python research/posttrain/assemble_tasks.py   # anchors, private rows, base + siblings
python common/sync.py --check                 # shared grader modules have not drifted
```
