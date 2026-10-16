# V2.4A — Approved analysis settings

This checkpoint updates the scientific configuration contract to schema version 2, without changing Streamlit controls or executing PyDESeq2.

The analysis configuration now includes `min_count` (default 10), `min_fraction` (default 0.20), requested `dispersion_fit` (`parametric`), and `refit_cooks` (`true`). All fields are fingerprinted. `filter_method: none_at_intake` remains explicit: filtering is a later DE-stage operation, not an intake modification. Runtime CPU allocation is recorded in the model manifest, but is not included in scientific approval.

**Current engine limitation:** The V2.1 PyDESeq2 wrapper fixes the requested dispersion fit to its default `parametric` and Cook refitting to `True`; schema v2 rejects other values instead of promising unsupported choices. Only the count thresholds are configurable at this stage.

**Important:** V2.4A adds an approval contract only. `app.py` still has no controls for the thresholds and does not execute differential expression. V2.4B must enforce approval again immediately before execution and pass approved filtering settings into `FilterPolicy`. A separate guard must ensure the same settings reach PyDESeq2. Existing benchmark configuration fingerprints use schema v1 and must not be reused as schema v2 approvals.

## Local verification

Run `python -m py_compile omicsgpt/vnext_workflow.py` and `python -m pytest -q tests/`. Confirm the UI intake still works without running DE. Review `git status --short` before staging the three files in this package.
