# Exploratory notebooks

These notebooks preserve the wider analyses and prior results. Run their root-setup cell first when opening them from this directory. Historical outputs can refer to old paths and sample definitions.

| Notebook | Purpose | Previous location |
|---|---|---|
| [Static geometry exploration](static_geometry_exploration.ipynb) | Broad geometry metrics, earlier mixed models, leave-one-out radial variants, plots | `01_static_geometric_metrics.ipynb` |
| [Dynamic microdynamics](dynamic_microdynamics.ipynb) | SDI, sustained-direction precision and micro-to-macro models | `02_dynamic_conversation_metrics.ipynb` |
| [PR alternatives](participation_ratio_alternatives.ipynb) | Bias-corrected PR, trajectories and animations | `03_participation_ratio_analysis.ipynb` |
| [Dynamic organization](dynamic_organization.ipynb) | Recurrence and order-based organization metrics | `dynamic_analysis.ipynb` |
| [Organization benchmark](dynamic_organization_benchmark.ipynb) | Ridge, ranking, HMM, language-model and GRU comparisons | Same filename at root |
| [Legacy metrics](legacy_metric_analysis.ipynb) | Earlier partial correlations, effect sizes and bootstrap analyses | `sub_analysis.ipynb` |

The static exploratory notebook retains its original cells, including a stray `zsx` scratch cell that must be skipped for sequential execution. Duplicate models and historical interpretations are retained as context, not endorsements. The clean paper preparation notebook excludes them.

Dynamic organization and its benchmark require `analysis_exports/turns_dynamic.csv`, `turn_embeddings.npy` and the matching manifest. The full historical/free-energy notebook at the root contains their export stages. They are not outputs of the new paper pipeline. Other exploratory notebooks calculate their own embeddings and can require additional model downloads/packages.

Most existing relative paths are preserved by setting the working directory to the repository root. Two erroneous `/first_approach/` absolute paths in the legacy metrics notebook were corrected. Exploratory notebooks may replace legacy exports, but do not write the new paper output directory.
