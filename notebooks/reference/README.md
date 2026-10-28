# Frozen notebook baseline

These are the pre-reorganization paper notebooks and their plotting helper.
They are intentionally independent of `analysis/` so the equivalence test is not
comparing the Python implementation to itself. `snapshot.json` hashes their exact
bytes. Do not edit them to make a failing comparison pass.

Run them through `python -m analysis.verify`; the verifier adapts only paths and
explicit config parameters in memory, then executes them in fresh kernels. They
are not supported entry points for producing the current paper exports directly.
The active, thin notebooks are in `../paper/`.
