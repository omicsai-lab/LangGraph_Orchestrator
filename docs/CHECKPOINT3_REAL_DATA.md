# Checkpoint 3 — real-data intake regression

This checkpoint adds **two tests** for the existing intake module. It does not change the module or Streamlit.

## Inputs

The tests expect `bc_counts_transposed_condensed.csv` and `bc_meta.csv` in the repository root. They use pandas CSV loading (`index_col=0` for counts), as the current application does. Strict CSV parsing and duplicate-header detection remain future work.

## Run

From the VS Code project root, in `.venv`:

```powershell
python -m pytest -q tests/test_intake_real_data.py
python -m pytest -q tests/
```

## What is checked

- Independent reference checks for file dimensions, sample IDs, missing/negative/fractional counts, and all-zero genes.
- Correct orientation and sample matching against the existing intake report.
- Identical prepared analytical matrices whether the uploaded counts are samples-by-genes or genes-by-samples.
- No mutation of either source DataFrame.

## Limitations

This does **not** test raw CSV byte preservation, duplicate CSV headers, scientific design/contrast selection, PyDESeq2, or Streamlit. The condensed test data contain 187 samples and 110 genes; they are not a performance benchmark for a full transcriptome.

## Acceptance criterion

Both new tests and the entire existing test suite pass locally. Confirm expected values before committing.
