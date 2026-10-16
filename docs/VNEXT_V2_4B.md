# V2.4B — Guarded statistical execution

`omicsgpt.guarded_execution.execute_approved_analysis` validates the researcher approval fingerprint, prepares the approved count matrix, rechecks metadata order and experimental design, constructs the approved `FilterPolicy`, and invokes only PyDESeq2. It never falls back to a different test. CPU count is execution provenance rather than a scientific setting.

Run `python -m pytest -q tests/test_guarded_execution.py` followed by `python -m pytest -q tests/`.

**Security/scientific limitation:** A fingerprint over a configuration is not a cryptographic seal of mutable in-memory DataFrames. The executor checks structure and design again, but does not independently rehash source bytes against the currently held DataFrames. For production execution, reconstruct the review bundle from the immutable upload bytes immediately before authorization, or extend the bundle with verified raw-byte provenance. Also, `engine=` is a dependency-injection seam for tests, not an API to expose to users.

**Next step:** real-data Docker execution through the guarded path and a numerical comparison against the existing benchmark; do not wire into Streamlit until complete.
