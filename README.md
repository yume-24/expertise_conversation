# Expertise and conversational geometry

A configured Python pipeline reproduces the paper's utterance embeddings, four
geometry metrics, mixed models, bootstrap tests and figures. Notebooks are optional
walkthroughs that call the same Python functions.

## Run

From the repository root, with Python 3.12 and uv installed:

```bash
uv sync --locked --inexact
uv run python -m analysis.run
```

`--inexact` preserves the extra packages in the existing exploratory environment
while installing the locked paper dependencies.

The default is `analysis/configs/paper.toml`. Outputs go to `outputs/paper/`.
A first run needs the pinned MiniLM model: set `local_files_only = false` in the
config to allow its initial download, then restore `true` for offline runs.
There are no automatic package/model installations inside the analysis code.

The existing project environment also works without uv:

```bash
.venv/bin/python -m analysis.run
```

Rerun individual stages without repeating the whole pipeline:

```bash
uv run python -m analysis.run --stage metrics
uv run python -m analysis.run --stage statistics
uv run python -m analysis.run --stage figures
```

`metrics` reuses embeddings only when ordered text/IDs, model revision, seed,
package versions and matrix hash match. Changed inputs invalidate the cache.
`statistics` and `figures` reject incompatible cached inputs. Use
`--rebuild-embeddings` to deliberately encode from scratch. Repeating a run
replaces that config's generated outputs. `--output-dir` selects an isolated run.

## Configs and definitions

| Config | Speaker-centroid utterances | Output directory |
|---|---|---|
| `analysis/configs/paper.toml` | At least five words, matching the current frozen notebooks | `outputs/paper/` |
| `analysis/configs/all_utterances.toml` | All nonempty utterances | `outputs/all_utterances/` |

```bash
uv run python -m analysis.run --config analysis/configs/all_utterances.toml
```

The reorganization intentionally preserves the current notebooks' five-word
centroid cutoff in the default. Earlier discussion favored all utterances, while
the checked-in notebooks still used five words; both definitions are now explicit
and tested independently. Choosing which to report is a methodological decision,
not a side effect of moving files. The working manuscript currently references
figures from the default run; its prose/table values are not automatically rewritten.

For both configs, median/Q90 radial distances use cosine distances to the normalized
pooled centroid of **all nonempty utterances**. PR is ordinary centered-covariance
participation ratio. The count covariate is total nonempty utterance count. Expertise
is coded 0–4; CIs are pointwise REML Wald intervals, p-values are ML parametric-bootstrap
likelihood-ratio tests, and BH/Holm corrections cover eight expertise tests.
The default uses seed 20260913 and 999 bootstrap replicates per test.

## Layout

```text
analysis/           Python modules, configs and regression tests
  legacy/           Retained historical code; not the supported paper pipeline
data/               Input transcripts; wired/ is the current paper corpus
notebooks/
  paper/            Thin walkthroughs calling analysis/
  reference/        Frozen original notebooks for independent comparison
  exploratory/      Retained research, including FEP; not validated as one pipeline
paper/              Working LaTeX source and bibliography; previous/ holds the old draft
outputs/
  paper/            Default run: embeddings/, tables/, figures/, manuscript/
  all_utterances/    Alternate run with the same output structure
  validation/       Independent notebook/Python runs and comparison reports
  figures/retained/  Existing manuscript assets without verified generators
  legacy/           Consolidated historical exports and visualizations
```

Each run saves its resolved config, transcript/model/package provenance and output
hashes. Figures are separate 300-dpi PNG and vector PDF files for each metric.
Key results are `tables/conversation_metrics.csv` and
`tables/statistics/paper_results_table.csv`; full standardized coefficients and
confidence intervals are in `tables/statistics/mixed_model_results.csv`.

## Verify against notebooks

```bash
uv run python -m unittest discover -s analysis/tests -v
uv run python -m analysis.verify
uv run python -m analysis.verify --config analysis/configs/all_utterances.toml
```

Verification runs the frozen original notebooks in fresh kernels and the Python
pipeline independently, with matching settings and separate outputs. It requires
**exact equality**, with no numerical tolerance, for the embedding matrix, utterance
table, metrics, model results, report table, sample audits and every bootstrap draw.
The all-utterance check changes the frozen notebook's selection parameter to zero;
it does not compare two different metric definitions.

Both configurations passed locally on 2026-10-28: 6,381 utterances, 105 conversations,
21 videos and eight tests with 999 draws each. Reports are in
`outputs/validation/<config>/comparison.json`. `--compare-only` rechecks saved runs.
PDF timestamps and model-summary dates are not compared. Exact numerical equality
was established in the local Python 3.12 environment; other hardware/BLAS builds
may introduce floating-point differences. `uv.lock` pins the dependencies, including
transitive versions used for this local validation.

## Manuscript

```bash
uv run python -m analysis.build_paper
```

Requires `pdflatex` and `bibtex` from a separately installed TeX distribution.
The PDF and build files go to `outputs/paper/manuscript/`. LaTeX references generated
plots directly; no copying into a second figures directory is needed.

The migration build succeeded, with two existing draft warnings: the citation key
`reimers-gurevych-2019-sentence` and figure reference `fig:participation-ratio`
are unresolved. Both were present before the reorganization.

Some historical illustrations and duplicate old result panels are still referenced
by the draft. They are retained in `outputs/figures/retained/` so this cleanup does
not silently change the manuscript. They are **not regenerated** by the new pipeline;
see that directory's README before deleting them. The full pipeline reproducibility
claim covers the current analysis and its generated figures, not every historical
research artifact. Transcript content was not edited during this reorganization.
The previously noted CRISPR quoting/segmentation issue remains a separate data audit.

See `analysis/migration.json` for old-to-new paths and the one byte-identical HTML
duplicate removed during consolidation. No unique input data or exploratory
research was deleted.
