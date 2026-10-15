# V2.2: independent PyDESeq2 benchmark

Run the provided script from the repository root in the pinned Docker image. This benchmark uses the breast-cancer example only, with an explicitly specified High-risk versus Average-risk contrast. It is not an interactive user approval or an end-to-end application validation.

Outputs are `differential_expression.csv` and `run_manifest.json` in the selected output directory. Results and timings are not precomputed. The source CSVs are never rewritten. Do not commit output files before reviewing their content and privacy implications.

The 10-count/20%-of-samples prefilter is a benchmark policy, not a universally validated default. Results require comparison to an independent R DESeq2 analysis before claiming numerical validation.
