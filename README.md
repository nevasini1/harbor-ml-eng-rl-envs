# Harbor ML-engineering RL environments

Spike and Harbor task work for a CPU-only agent-evaluation environment: adapt a pretrained molecular model under a fixed compute budget, scored by held-out ROC-AUC with anti-substitution checks.

## Layout

| Path | What |
|---|---|
| [`spike/`](spike/) | Measurement scripts, private-split tooling, verifier prototype, and [`SPIKE_RESULTS.md`](spike/SPIKE_RESULTS.md) |
| [`tasks/mol-property-adapt/`](tasks/mol-property-adapt/) | Harbor task (ChemBERTa + MoleculeNet / Tox21 region holdout) |
| [`sciml-protein-regression/`](sciml-protein-regression/) | Earlier protein-track Harbor scaffold (abandoned after spike) |
| [`jobs/`](jobs/) | Harbor run logs from local trials |

## Spike verdict (short)

- **Protein / FLIP2 Hydro:** Gate A failed — combinatorial core library; one-hot ridge matches or beats ESM.
- **Protein / Meltome:** Gate B failed — ~85 min per CPU embedding pass.
- **Molecular / ChemBERTa + Tox21 region holdout:** both gates passed; this is the active track.

Full writeup: [`spike/SPIKE_RESULTS.md`](spike/SPIKE_RESULTS.md).

## Note on private holdout

This public repo includes `spike/PRIVATE_SEED` and `spike/split/private/` by request. That publishes the graded holdout; do not treat the reward as un-gameable if agents can read this repository.

## Regenerate data / caches

```bash
cd spike
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas scipy numpy scikit-learn torch transformers safetensors huggingface_hub rdkit
.venv/bin/python fetch_flip2.py          # FLIP2 pins
.venv/bin/python moleculenet.py --help   # MoleculeNet downloads
```

Third-party clones (`.research/`, `_research/`), the spike venv, embedding caches, and large fixture checkpoints are gitignored.
