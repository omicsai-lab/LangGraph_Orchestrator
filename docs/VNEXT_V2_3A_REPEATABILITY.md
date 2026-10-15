# OmicsGPT vNext V2.3A — PyDESeq2 repeatability

Run the existing benchmark twice in **different output directories** with identical source data, filter policy, contrast, and CPU settings. Then use `scripts.check_repeatability` to compare both runs. The comparator checks source/config hashes, model settings, filter audit, gene IDs, all six DE numeric columns, and significance/direction stability. It uses absolute tolerance 1e-8 and relative tolerance 1e-6, and separately reports exact equality. A successful comparison does **not** establish statistical validity; an independent R DESeq2 comparison remains necessary.

The comparison does not rerun statistical estimation and does not modify either benchmark output directory. The JSON report can be written under ignored `benchmark_outputs/`.
