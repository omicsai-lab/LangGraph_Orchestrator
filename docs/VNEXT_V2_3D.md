# V2.3D: PyDESeq2 model provenance

This checkpoint adds read-only fitted-model diagnostics to the existing statistical engine return dictionary and benchmark manifest. It does not change fitting, filtering, or contrast selection.

Fields include requested versus actual dispersion fit (from `dds.fit_type` and `dds.uns['disp_function_type']`), Cook's refit setting, and ordered gene identifiers flagged in `dds.var['replaced']` and `dds.var['refitted']` when present. Missing optional flags are recorded as JSON `null`, not zero. Missing essential diagnostics stop execution.

## Verify

1. Run `python -m pytest -q tests/` in the local virtual environment.
2. Re-run the benchmark inside the existing Docker image, with a **new** output directory `benchmark_outputs/v2_3d` (do not overwrite V2.2).
3. Inspect `benchmark_outputs/v2_3d/run_manifest.json`, including the `diagnostics` object.
4. Compare the new `differential_expression.csv` to the V2.2 output using `scripts.check_repeatability` after checking its provenance compatibility. A manifest difference due to the added diagnostics field is expected; compare numeric tables separately if needed.
5. Verify whether the replaced genes match the eight genes from the independent R DESeq2 run. A matching count alone is insufficient.

The `refitted` column is optional and is distinct from the `replaced` flag. This extractor targets PyDESeq2 0.5.4 internals; review it before upgrading PyDESeq2.
