# Checkpoint 4B.1 — read-only Streamlit intake preview

**Scope:** display the existing strict CSV and count-matrix inspection outputs. No statistical execution changes.

## Integration into `ultimate_agent.py`

1. Add this import with the other imports near the top:

```python
from omicsgpt.intake_panel import render_read_only_intake
```

2. Find the `with col1:` block under `col1, col2 = st.columns([1, 2])`. Immediately after the two file-uploaders (`counts_file = ...` and `metadata_file = ...`), add:

```python
    with st.expander("Inspect uploaded data (read-only preview)", expanded=False):
        render_read_only_intake(counts_file, metadata_file)
```

**Do not replace** the downstream `pd.read_csv`, DE, or session-state logic in this checkpoint. This is a deliberately non-gating preview; the old computation is still unsafe and must be addressed in 4B.2/4B.3 before calling the intake workflow validated.

## Smoke test

- Run `python -m pytest -q tests/` (existing suite should remain at 32 passing).
- Start Streamlit only after installing its full app dependencies in a compatible environment; verify the preview using both example CSVs.
- Confirm that the panel displays 187 samples, 110 genes, 187 matching IDs, three all-zero genes, and the SHA-256 hashes.
- Upload malformed CSVs and confirm that errors are shown rather than silent repairs.
- Confirm existing DE computation is unchanged; **do not use its output for scientific conclusions** until the gating and statistical safety changes are complete.

## Known limitations

- This implementation loads entire CSVs into memory to validate them; the preview itself is small. A resource/size limit and chunked validation are future work.
- The strict reader assumes the first count-file column holds row identifiers; users with other layouts require a later explicit selector.
- Sample-ID selection is for preview only and does not configure the legacy statistical engine.
- There is no upload-specific state invalidation or confirmation gating yet.
