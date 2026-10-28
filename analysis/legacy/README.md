# Historical source

These scripts are preserved for research provenance, not used by the current
paper pipeline. `main.py` is the old expert/novice text-file graph workflow;
`first_approach/` contains the earlier embedding/graph/classification work.
`run_paper_notebooks.py` is the retired runner, replaced by `python -m analysis.run`.

Imports were adjusted to the new package locations where directly referenced.
Historical scripts can still contain hard-coded paths and additional dependencies;
they have not been validated as a reproducible workflow. Inputs remain in `data/`,
and prior outputs are in `outputs/legacy/`. Do not substitute those results for
config-generated results without explicitly reconciling definitions and inputs.
