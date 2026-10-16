# V2.4C: Streamlit approved analysis integration

Adds filtering controls, fingerprint-based researcher confirmation, guarded PyDESeq2 execution, results and provenance downloads. The UI calls `execute_approved_analysis`, not the statistical engine directly. `omicsgpt/run_state.py` isolates state transitions for fast tests.

## Acceptance checks
1. Run `python -m py_compile app.py omicsgpt/run_state.py` and `python -m pytest -q tests/`.
2. Launch the existing Docker image with the mounted repository and `streamlit run /app/app.py`.
3. Upload the breast cancer counts and metadata, confirm sample ID `Sample`, grouping `condition`, numerator `risk category: High`, reference `risk category: Average`.
4. Approve settings (minimum 10 counts, 20% of samples), run the model, and verify 75 retained genes, mean dispersion fallback, and eight refitted genes.
5. Change the filtering threshold or contrast: approval and old results must disappear. Restore previous values: approval must still be required.
6. Do not commit until the UI and tests are verified. `ultimate_agent.py` and deployed Docker release are unchanged.

## Known limitations
- Session-state results are transient and not a durable run store.
- Source bytes are re-read on each Streamlit rerun, but the guarded executor does not yet independently re-hash mutable in-memory DataFrames; a hardened immutable input handoff remains future work.
- Model warnings appear in Docker logs; a structured warning audit is future work.
- This UI supports only the current two-group bulk RNA-seq workflow, categorical covariates, requested parametric fit, and Cook refitting enabled.
