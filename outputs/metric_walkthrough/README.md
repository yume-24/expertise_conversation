# Metric walkthrough figures

Illustrations used to explain the conversation metrics belong here.

- `paper/` contains the exact time-series and speaker-centroid images referenced
  by the working manuscript, plus the available time-series PDF.
- `wired_gravity_12/`, `wired_gravity_14/` and `wired_gravity_16/` contain the
  conversation-specific walkthrough exports.

The manuscript images differ from the conversation-specific exports and are
preserved separately. The recovered generator is
`analysis/visualize_metric_walkthrough.py`. Run from the repository root after
generating cached embeddings:

```bash
uv run python -m analysis.visualize_metric_walkthrough --conversation wired_gravity_14 --static-only
```

Omit `--static-only` to include animated playback in the HTML walkthrough. The
script reads `outputs/paper/embeddings/` by default and writes the conversation's
subdirectory here; it does not overwrite the curated images in `paper/`.
Use `--output-dir` for a separate export location, or `--list` to list conversations.

This walkthrough includes all nonempty utterances in speaker centroids, matching
the `all_utterances` analysis definition. Its PCA rings show projected Euclidean
distance quantiles for illustration; reported radial metrics use cosine distances
in the original embedding space.

The script was recovered from commit `6752eb3`, which was left outside the current
branch during the October 16 rebase. See `analysis/walkthrough_recovery.json` for
its source and path updates. It runs separately from `python -m analysis.run`.
Recovery does not establish pixel-for-pixel equivalence with the curated paper
images, which may include subsequent edits.
