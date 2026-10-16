# OmicsGPT V2.5A — deterministic results exploration

This checkpoint adds only `omicsgpt/results_explorer.py` and unit tests. It does not change PyDESeq2, approval, Streamlit, or any uploaded data.

## Candidate selection

- `balanced`: up to `n` positive and `n` negative log2FC genes, ordered by effect magnitude, then adjusted p-value, then gene ID.
- `significance`: up to `n` genes ordered by adjusted p-value then gene ID.
- `effect`: up to `n` genes ordered by absolute log2FC then adjusted p-value then gene ID.
- `manual`: explicit IDs, retaining the researcher's order and deduplicating. Manual selection does not impose automatic FDR criteria.
- Automatic selection requires finite log2FC and adjusted p-value at or below the selected FDR threshold; optional minimum absolute log2FC is applied. Missing values are not imputed.
- These are **exploratory candidates**, not validated biomarkers. No biological evidence is used yet.

## Volcano data

`prepare_volcano` preserves the original p-values and returns `volcano_y` and `volcano_display_capped`. Exact zero p-values are capped **for display only** at one-tenth the smallest observed positive p-value, if one exists. If there are no positive p-values, zeros remain unplottable and are counted. Missing data are reported in an audit. The axis may be raw p-value or adjusted p-value; FDR significance always uses `padj`.

## Next checkpoint

V2.5B will connect these pure functions to a Plotly volcano plot, interactive selection controls, a searchable results table, and plain-language help. Changing exploration settings must never rerun DE or invalidate scientific approval.
