# OmicsGPT V2.6A.3 — Audited benchmark preparation

This **benchmark-only** utility excludes exactly three unresolved gene-label columns named `TRNAV-CAC` from the full breast cancer counts CSV. It does not alter the original file, repair annotations, perform DE, or contact external services.

## Scientific rationale and limits

The three original CSV field positions (zero-based, including the first sample-ID field) are 2031, 2032, and 2152. They contain different count profiles, so they cannot be assumed to be duplicate measurements of one biological feature. Under the benchmark filter of at least 10 counts in at least 20% of samples, all three fail eligibility. The utility verifies this against the actual file and verifies that the retained gene set size is unchanged. The resulting derived dataset **must not** be reused under different filtering settings without revalidation. Raw-count provenance remains provisional.

## Execution from repository root

```powershell
python -m pytest -q tests/test_prepare_full_benchmark.py
python -m pytest -q tests/
python -m scripts.prepare_full_benchmark `
  --counts bc_counts_transposed.csv `
  --metadata bc_meta.csv `
  --outdir benchmark_outputs/v2_6a3_prepared
```

Then run the existing non-fitting audit against the derived matrix:

```powershell
python -m scripts.audit_full_counts `
  --counts benchmark_outputs/v2_6a3_prepared/counts_unique_features.csv `
  --metadata bc_meta.csv `
  --outdir benchmark_outputs/v2_6a3_intake
```

The preparation manifest records source and derived SHA-256 hashes, positions, maxima, sample counts, and filtering decisions. Both output directories must be new. No output is overwritten. The script uses a streaming CSV reader and does not load the entire matrix into pandas.

## Interpretation

A passing benchmark establishes structural compatibility with the current intake contract, **not** that the excluded columns have been biologically resolved. Future general-purpose duplicate-feature handling requires researcher review and source annotation provenance.
