# V2.6A.5 — Full-data audit stabilization

## Scope

- Fixes the remaining string-versus-integer comparison in the **audit script only**.
- Converts strictly validated decimal count strings into a numeric NumPy matrix once, then uses the **same matrix** for library-size totals, filtering, and all-zero gene counts.
- Preserves input bytes, existing hashes, filtering thresholds, and benchmark-only status.
- Adds per-stage timings to help locate the cost of full-transcriptome validation.
- Adds end-to-end CLI tests with an unnamed identifier header and sample-by-gene counts, plus invalid-input tests.
- Does not modify the statistical engine, intake contracts, or Streamlit app.

## Interpretation and performance

The derived full benchmark matrix excludes three ambiguous TRNAV-CAC features under a separately documented benchmark-only policy. This is not an annotation-resolution rule for future datasets.

This change is primarily a correctness fix, **not** a claim of faster import. The current strict CSV intake still materializes the full matrix and can be expensive; inspect stage timings before attempting optimization. No PyDESeq2 fitting is performed by this audit.

The source matrix is provisionally treated as raw counts; upstream quantification provenance is not yet confirmed.

## Verification

1. `python -m py_compile scripts/audit_full_counts.py`
2. `python -m pytest -q tests/test_full_audit_integration.py`
3. `python -m pytest -q tests/`
4. Once, with a **new** output directory:

```powershell
Measure-Command {
    python -m scripts.audit_full_counts `
      --counts benchmark_outputs/v2_6a3_prepared/counts_unique_features.csv `
      --metadata bc_meta.csv `
      --outdir benchmark_outputs/v2_6a5_intake
}
```

Expected from earlier structural checks: 187 samples, 39,373 gene columns, 22,470 retained at min_count=10 and min_fraction=0.2. These are **expectations to verify**, not hardcoded values in the script.

Inspect `stage_timings_seconds`, `library_sizes`, `filter`, and both source hashes in `intake_audit.json`. Do not commit until the full audit succeeds.
