# Expertise and conversational geometry

The paper examines how conversational geometry varies with partner expertise within matched WIRED videos. The main workflow now has two notebooks; exploratory analyses are separate.

## Main paper workflow

Only the notebooks in `notebooks/paper/` form the analysis workflow
used for this paper. 

The notebooks in `notebooks/exploratory/` and the free-energy notebook
are retained as supplementary development work and are not required
to reproduce the paper’s reported analyses.

Run in this order, using a fresh kernel for each notebook:

1. [Prepare the four metrics](notebooks/paper/01_static_geometric_metrics.ipynb): load transcripts, embed utterances, calculate geometry, validate the sample, and export the metrics and descriptive plots.
2. [Statistical significance](notebooks/paper/02_statistical_significance.ipynb): fit video random-intercept models with and without log utterance count; calculate bootstrap expertise p-values, multiple-testing adjustments, effect sizes and diagnostics.

The notebooks locate the repository from either its root or their own directory. No shared kernel state is needed.

```bash
# Existing project environment, Python 3.12:
.venv/bin/python scripts/run_paper_notebooks.py

# Reuse an existing paper metric export:
.venv/bin/python scripts/run_paper_notebooks.py --only statistics
```

For a new environment, install [requirements-paper.txt](requirements-paper.txt). These pin the direct packages used locally; they are not a complete cross-platform dependency lock. Notebook 01 defaults to local-only loading of a pinned MiniLM revision. If that model is absent, set `LOCAL_FILES_ONLY = False` in its embedding setup cell for the first download.

### Inputs and outputs

```text
data/wired/**/*.csv
    → notebooks/paper/01_static_geometric_metrics.ipynb
    → outputs/paper/conversation_metrics.csv
    → notebooks/paper/02_statistical_significance.ipynb
    → outputs/paper/statistical_significance/paper_results_table.csv
```

`outputs/paper/` also contains the utterance table, embedding matrix, descriptive plots and a manifest of transcript hashes, model revision and metric definitions. The statistics subfolder contains diagnostics, bootstrap draws and its own source hash and settings. Rerunning replaces these generated files.

Historical `analysis_exports/`, `outputs/statistical_significance/` and manuscript figures in `paper/figures/` are retained. They are not regenerated or silently overwritten by this workflow. Use the new paper output directory for newly generated results; manuscript figures still need to be reconciled with the final metric choices.

### Current metric definitions 

| Outcome | Utterances used | Definition |
|---|---|---|
| Speaker-centroid distance | At least five words | Cosine distance between expert and partner mean embeddings |
| Median radial distance | All nonempty utterances | Median cosine distance to normalized pooled centroid |
| Q90 radial distance | All nonempty utterances | 90th percentile of the same distances |
| Participation ratio | All nonempty utterances | Ordinary centered-covariance PR |

The count adjustment uses **total nonempty utterances**, including for the centroid model. These existing choices were preserved during reorganization. They do not constitute a harmonized utterance-selection policy. Corrected PR and leave-one-out radial distances remain alternatives, not interchangeable versions of these outcomes.

The statistical bootstrap tests **expertise**, not the improvement from adding count. CIs are approximate REML Wald intervals; p-values use ML parametric-bootstrap likelihood-ratio tests. BH and Holm adjustments cover eight expertise tests. Residual assumptions and the common linear slope still require assessment. These are exploratory analyses, not newly preregistered tests.

## Exploratory and historical analyses

See [the exploratory notebook guide](notebooks/exploratory/README.md). These notebooks retain existing analysis cells and historical outputs. They are not part of the paper run order and have not all been validated as clean-kernel pipelines.

## Free-energy work

[big_analysis_fep_implementation.ipynb](big_analysis_fep_implementation.ipynb) remains **unchanged at the repository root**, including all its supporting analyses. It is separate from the paper workflow and may write legacy files under `analysis_exports/`. Moving or pruning it was deliberately excluded from this reorganization.

## Reorganization validation (2026-09-15)

Both paper notebooks completed in fresh kernels from their new directory: 105 conversations across 21 videos, eight converged mixed models, and 999 bootstrap replicates per expertise test. Metric definitions were retained, and historical outputs were not overwritten.

The transcripts were revised before the notebook reorganization, including changes to utterance boundaries. All 105 historical conversation counts match the transcript version in commit `38b08b9` (2026-08-14). The revised transcripts in commit `dd914b9` (2026-08-23) account for the different utterance counts in 12 conversations. These differences reflect transcript revisions, not utterances lost during notebook reorganization. Historical exports represent the earlier transcript version; the rebuilt paper outputs use the revised transcripts.

The comparison is saved in `outputs/paper/reorganization_comparison.csv`. The BH-significance pattern is unchanged: centroid distance and PR pass in both specifications, median radial distance only with count adjustment, and q90 only without adjustment. Numerical estimates and some p-values differ. Use the new manifests to identify precisely which inputs underlie future results.

Outstanding transcript check: `data/wired/wired_crispr/wired_crispr_16.csv` has a mismatched closing quote on physical line 6 that causes the CSV parser to absorb the next speaker row into the same utterance. Physical lines 13–14 also contain repeated wording that should be checked against the source video. These issues have not yet been corrected; resolve them and rerun both paper notebooks before final reporting. The validation results above describe the inputs before those corrections.

## Plot styling

[plot_style.py](plot_style.py) is the shared source for fonts, grid/spine defaults, metric labels, expertise colors and paper export settings. Both paper notebooks use the same trajectory plotting function. Unadjusted models use teal circles; count-adjusted models use orange squares. Expertise colors are keyed to the level, so reversing or filtering the levels does not reassign their colors.

Paper figures are saved as 300-dpi PNG and vector PDF. Their uncertainty labels retain the actual statistical meaning: descriptive trajectories have no confidence band, while model-effect plots show pointwise 95% Wald intervals. Styling does not change estimates, data, axis transformations or uncertainty calculations.

Exploratory notebooks import the same defaults and use the shared expertise palettes where applicable. Old embedded static pictures were cleared; rerun their plot cells after the required preparation to regenerate them. Specialized correlation heatmaps, animations and interactive plots retain their task-specific encodings. Existing manuscript images and the free-energy notebook remain unchanged.
