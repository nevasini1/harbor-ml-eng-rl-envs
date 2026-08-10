# mleval/sciml-protein-regression

Harbor task (1 GPU, 4h): fine-tune `facebook/esm2_t6_8M_UR50D` on a Meltome thermostability split; score Spearman on a private held-out set.

## Layout

```
task.toml
instruction.md
environment/          # agent image (base model + /data/train.csv.gz)
solution/solve.sh     # oracle (frozen backbone + trained head)
tests/                # separate verifier image (private_test + grade.py)
scripts/make_private_split.py
```

## Regenerate private split

```bash
python scripts/make_private_split.py
```

The seed lives only in that script (not in the agent image).

## Run

```bash
# oracle smoke
harbor run -p ./sciml-protein-regression -a oracle

# take-home agent smoke
export OPENAI_API_KEY=...   # set locally; do not commit
harbor run -p ./sciml-protein-regression -a codex -m openai/gpt-5.6-sol -n 1
```

## Notes

- Hydro was measured in `../research/` and failed the headroom gate (combinatorial library; one-hot saturates / WT-transfer fine-tune does not climb). This task uses Meltome-mixed with a fresh unpublished resplit.
- Harbor's local Docker provider rejects `network_mode="no-network"`. This task uses `network_mode="public"` plus a **separate verifier** with `private_test` baked into the verifier image. Use Modal/E2B if you need provider-enforced egress deny.
- `reward.json` values must all be numeric (Harbor `VerifierResult`); string fields go to `reward_meta.json`.
- Reward is tiered (`0` / `0.5` / `1.0`) from Spearman vs fixed `T_weak` / `T_strong` in `tests/tiers.json` (calibrated offline from a frozen ESM probe and a strong-oracle target).
- Re-calibrate locally: `python scripts/calibrate_tiers.py` (writes `tests/tiers.json`).
