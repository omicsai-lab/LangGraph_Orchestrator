# V2.3C: Dispersion sensitivity and outlier diagnostics

This is a **secondary sensitivity analysis**; the primary R DESeq2 reference and PyDESeq2 benchmarks are unchanged.

Run from repository root in RStudio: `source("validation/deseq2_mean_sensitivity.R")`.

This script independently reads the original count and metadata CSVs, applies the same >=10 counts in >=ceil(0.2*n) samples rule, and fits DESeq2 1.46.0 with `fitType="mean"`, using the same explicit High vs Average contrast and default Cook's refitting behavior as the primary R reference. Outputs are written to `benchmark_outputs/v2_3c_r_mean/`, preserving the earlier runs.

Compare with the existing comparator (from repository root):

```powershell
python -m scripts.compare_r_deseq2 --python-dir benchmark_outputs/v2_2 --r-dir benchmark_outputs/v2_3c_r_mean --output benchmark_outputs/v2_3c_comparison_mean.json
```

Compare the new JSON against `benchmark_outputs/v2_3b_comparison.json`, especially LFC standard errors, Wald statistics, p-values, and FDR decisions. Do not relax numerical tolerances to force a pass. The original R local-fit reference remains the primary independent comparison.

**Python diagnostics gap:** the existing Python manifest reports the number of Cook's outlier replacements only in console logs. Do not infer their identities from the R results. Capturing the actual Python dispersion fit and replaced genes requires a separate, tested change to `run_pydeseq2` and the benchmark manifest; do not claim these fields are already available.
