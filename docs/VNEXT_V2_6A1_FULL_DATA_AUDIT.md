# OmicsGPT V2.6A.1 — Full-transcriptome intake audit

This checkpoint adds a non-fitting audit of full-transcriptome input using the existing strict OmicsGPT intake, approval fingerprint, and matrix preparation code. It does not change Streamlit, filtering, or the statistical engine.

## Methodological boundaries

- Treat counts as **provisionally raw** pending upstream quantification documentation. Integer values alone cannot establish provenance.
- Preserve original source bytes and SHA-256 hashes.
- Verify sample orientation, sample matching, count validity, and experimental design through the existing workflow.
- Summarize per-sample total counts as a descriptive library-size diagnostic, **not** as a normalization step.
- Calculate the predefined low-count filter (10 counts in at least 20% of samples) without removing genes from the original matrix.
- The script's programmatic approval fingerprint is **benchmark-only** and is not a substitute for researcher approval in the app.
- Output files are local, potentially sensitive research artifacts; do not commit them to Git.

## Execute

From the repository root:

```powershell
python -m scripts.audit_full_counts `
  --counts bc_counts_transposed.csv `
  --metadata bc_meta.csv `
  --outdir benchmark_outputs/v2_6a1_intake
```

The audit writes `intake_audit.json` and `sample_library_sizes.csv` in the new output directory and refuses to overwrite an existing nonempty output directory.

## Performance follow-up (separate checkpoint)

Only after inspecting the audit, benchmark the full-data PyDESeq2 model with an explicit memory limit, one CPU, and resource monitoring. Do not assume the condensed-dataset runtime scales linearly to 22,470 genes. Keep the statistical benchmark noninteractive and separate from the Fast/Deep Dive evidence retrieval modes.
