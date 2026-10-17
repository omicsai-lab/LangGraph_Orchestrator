# V2.6A.2 — Explicit unnamed row-identifier import

## Scope

A CSV with a **literally empty first header field** (e.g. `,GeneA,GeneB`) may be read **only** with `use_identifier_as_index=True` and without an alternate explicit identifier column. The first column is then interpreted as row identifiers. Its original header remains the empty string in `ImportedCSV.identifier_column`; no source bytes are changed. The source SHA-256 continues to cover the original upload.

Blank headers in any other position, whitespace-only first headers, duplicate headers, duplicate or missing row identifiers, malformed rows, and metadata imported without index mode continue to fail closed. The change does not perform gene identifier mapping, infer omics modality, or automatically decide the biological orientation. Existing metadata-based orientation validation and researcher approval remain required.

## Verification

Run `python -m py_compile omicsgpt/csv_import.py`, `python -m pytest -q tests/test_csv_unnamed_identifier.py`, and `python -m pytest -q tests/`. Then rerun `python -m scripts.audit_full_counts --counts bc_counts_transposed.csv --metadata bc_meta.csv --outdir benchmark_outputs/v2_6a1_intake`.

This patch changes only the importer and adds tests/documentation. It does not alter statistical fitting or pathway analysis.
