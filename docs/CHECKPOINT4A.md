# Checkpoint 4A — CSV import and experimental design

## Implemented (independent of Streamlit)

- `omicsgpt/csv_import.py`: strict UTF-8 CSV import, duplicate-header and duplicate-ID checks, rectangular rows, string preservation, and SHA-256 source hash. Values are not silently type-inferred, rounded, or filled. Counts and metadata can be imported independently.
- `omicsgpt/design.py`: explicit numerator/reference and categorical additive covariates; no automatic group selection, level dropping, or missing-value deletion. Full rank and residual degrees of freedom are checked, but **do not guarantee a scientifically appropriate model**.
- `omicsgpt/intake_review.py`: structured, JSON-serializable preview, dimensions, status, proposed transformations, and researcher-confirmation flag. It does not execute those transformations.

## Deliberate limitations

1. This is **not integrated** with Streamlit, PyDESeq2, or Docker.
2. CSV parsing is intentionally strict; alternative delimiters, non-UTF-8 encodings, sparse formats, and ID mapping are future work.
3. The two-group design validator requires exactly two observed groups; subsetting a larger cohort must be an explicit, separately logged operation.
4. Covariates are treated as categorical. Continuous covariates, paired/repeated measures, interactions, nesting, and nonindependence require separately designed workflows.
5. Minimum group size of two is a computational safeguard, not a power or replication adequacy guarantee.
6. The preview is limited to 5 rows and 6 count columns; **all** input values still require validation.
7. Raw-byte preservation currently means the loader computes a SHA-256 digest of uploaded bytes. Persisting the raw bytes securely and an immutable run manifest will be addressed in the provenance checkpoint.
8. Source files should not contain protected health information in filenames or sample IDs. Test files included here are example datasets.

## Run

From the repository root in the Python 3.11 `.venv` with pandas, NumPy, pytest:

```powershell
python -m pytest -q tests/
```

## Installation into existing repo

Copy only the **three new modules** (`csv_import.py`, `design.py`, `intake_review.py`), `tests/test_checkpoint4a.py`, and this document to matching locations in your existing `review-revision` checkout. **Do not overwrite** `intake.py`, `__init__.py`, existing tests, example CSVs, or application code.

Review `git status` and run the suite before committing.
