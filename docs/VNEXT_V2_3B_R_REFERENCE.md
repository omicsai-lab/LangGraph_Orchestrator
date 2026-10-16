# V2.3B — Independent R DESeq2 reference

## Scope

This is a **cross-implementation statistical comparison**, not a test that R and Python must agree exactly. It uses the original CSVs, not the Python differential-expression results, to independently build and fit the R model.

- R: 4.4.3; DESeq2: 1.46.0 (verify via generated session_info.txt).
- Python benchmark: PyDESeq2 0.5.4, previously run with 187 samples.
- Two-group design: `~ condition`, numerator `risk category: High`, reference `risk category: Average`.
- Prefilter: >=10 counts in >=ceiling(0.20 * n_samples), expected 38/187.
- DESeq2: `fitType="parametric"` requested, with DESeq2's internal fallback permitted and recorded; `minReplicatesForReplace=7`, `cooksCutoff=TRUE`, `independentFiltering=TRUE`.
- No Python outputs are read by the R fitting script.

## Run from the repository root in RStudio

```r
source("validation/deseq2_reference.R")
```

Or in PowerShell (if Rscript is on PATH):

```powershell
Rscript validation/deseq2_reference.R
```

Output: `benchmark_outputs/v2_3b_r_reference/` containing `deseq2_results.csv`, `filtering_audit.csv`, `model_diagnostics.json`, `session_info.txt`.

## Compare in the existing Python venv

```powershell
python -m scripts.compare_r_deseq2 --python-dir benchmark_outputs/v2_2 --r-dir benchmark_outputs/v2_3b_r_reference --output benchmark_outputs/v2_3b_comparison.json
```

This produces per-column paired finite counts, missingness mismatches, maximum and median absolute differences, tolerance counts, correlations, direction disagreements and FDR classification disagreements. Numerical tolerances are *diagnostic*, not pass/fail thresholds for method validity. A filter-gene-set mismatch stops the comparator.

## Interpret carefully

The 110-gene condensed input (75 genes after filtering) is a small, nonrepresentative feature set for dispersion trend estimation. A mismatch between R and Python may be due to dispersion-trend fallback, optimization, Cook's outlier refitting, or independent filtering. Inspect the R fit method and replaced genes before changing any parameters. Use a full-transcriptome dataset for the next validation stage. Generated benchmark outputs are ignored by Git; retain them locally or in a controlled analysis archive.
