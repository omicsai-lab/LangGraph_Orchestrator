# OmicsGPT V2.6A.4 — Library-size audit type compatibility

The strict CSV importer preserves input values as strings. The full-transcriptome
intake produces an approved counts matrix that may retain decimal integer strings,
so the audit must validate their values rather than require pandas integer dtypes.

The revised `summarize_library_sizes` accepts nonnegative int64-range integers and
exact unsigned decimal integer strings. It rejects floating-point objects, decimal
fractions, scientific notation, signs, whitespace, missing values, booleans,
negative counts, and overflow. No rounding, imputation, or source-data mutation
occurs. Leading-zero count strings are interpreted numerically after validation.

Per-sample sums use Python integers to avoid int64 accumulation overflow. The
existing benchmark-only approval, filtering policy, source hashes, and reporting
schema remain unchanged. This is not an upstream raw-count provenance check.

## Verification

1. Run `python -m pytest -q tests/test_full_counts_audit_types.py`.
2. Run `python -m pytest -q tests/`.
3. Rerun the full audit using a **new output directory** because earlier output
   artifacts are intentionally not overwritten.
4. Record elapsed time; optimize only after profiling. Do not repeatedly run the
   full dataset to test minor changes.
