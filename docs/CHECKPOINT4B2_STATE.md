# Checkpoint 4B.2a — confirmation-state contract

This checkpoint adds **only** a pure-Python confirmation fingerprint and unit tests. It does **not** change `ultimate_agent.py`, enable confirmation controls, or gate the legacy statistical analysis. The current Streamlit workflow remains unsafe for scientific inference until gating and statistical fallback changes are implemented.

## Scientific configuration bound to approval

- SHA-256 hashes of both uploaded source files
- resolved orientation and sample-ID column
- **ordered** sample IDs (order matters)
- grouping variable, numerator, reference, and ordered covariates
- **ordered** proposed transformations

The fingerprint uses canonical JSON serialization and SHA-256. A missing or invalid configuration fails closed. UI-only display preferences are deliberately excluded.

## Known limitations and next integration

- This helper does not independently verify the hashes, design validity, or transformations. Call existing intake/design validators first.
- Before wiring into Streamlit, add tests for file changes, controls changing during reruns, confirmation invalidation, and rejection of stale downstream results.
- The eventual analysis button must use the same validated, confirmed prepared matrix/configuration used to compute the fingerprint; do not reparse a different file or recompute the contrast from row order.
- If a new computational parameter affects results (filtering, engine, thresholds, model formula), add it to a **separate run configuration fingerprint** or extend this schema explicitly; never allow silent changes.
- Do not store source bytes in this fingerprint; store and protect source hashes and a separately versioned run manifest.

## Local verification

Run `python -m pytest -q tests/` and `python -m py_compile omicsgpt/confirmation.py`. Stage only these three new files. Review results before committing.
