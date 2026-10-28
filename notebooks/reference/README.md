# Frozen notebook baseline

These are the pre-reorganization paper notebooks and their plotting helper.
They are intentionally independent of `analysis/` so the equivalence test is not
comparing the Python implementation to itself. `snapshot.json` hashes their exact
bytes. Do not edit them to make a failing comparison pass.

Run them through `python -m analysis.verify`; the verifier adapts only paths and
explicit config parameters in memory, then executes them in fresh kernels. They
are not supported entry points for producing the current paper exports directly.
The active, thin notebooks are in `../paper/`.

## Run the comparison

From the repository root:

```bash
uv run python -m analysis.verify
uv run python -m analysis.verify --config analysis/configs/all_utterances.toml
```

Verification executes these notebooks in fresh kernels and the Python pipeline
independently, with matching settings and separate outputs. It requires exact
equality, with no numerical tolerance, for the embedding matrix and eight tables:
utterances, conversation metrics, analysis data, excluded conversations, video
coverage, mixed-model results, the paper results table and every bootstrap draw.
For the all-utterance config, the verifier changes the notebook selection cutoff
to zero in memory; it does not compare different metric definitions.

Both configurations passed locally on 2026-10-28 with 6,381 utterances,
105 conversations, 21 videos and eight tests with 999 draws each. Reports are in
`outputs/validation/<config>/comparison.json`; `--compare-only` rechecks saved
runs. A compact record is in `analysis/migration_validation.json`.

PDF timestamps and model-summary dates are not compared. Exact numerical equality
was established in the local Python 3.12 environment with dependencies pinned by
`uv.lock`; other hardware or BLAS builds may introduce floating-point differences.

## Migration scope

The default config preserves the original five-word minimum for speaker-centroid
calculations. The all-utterance config makes the alternative definition explicit.
Transcript content was not changed during migration; the previously noted CRISPR
quoting/segmentation issue remains a separate data audit. Historical research was
retained, with moves and the removal of one byte-identical HTML duplicate recorded
in `analysis/migration.json`.
