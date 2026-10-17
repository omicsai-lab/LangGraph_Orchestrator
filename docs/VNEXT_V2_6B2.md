# V2.6B.2 — Preranked GSEApy execution

Adds `omicsgpt/gsea_execution.py`, consuming the **unchanged** `GSEAInput` from V2.6B.1. Does not modify PyDESeq2, Streamlit, approval, or source datasets. Requires GSEApy (verified by researcher in Docker: 1.2.1).

- Ranking: full approved DE results, signed Wald statistic, not a candidate shortlist.
- Gene sets: locally loaded, validated GMT; explicit release and SHA-256 from input contract.
- GSEApy: prerank, permutation count 1000, seed 42, 1 thread, weight 1.0 by default. No network retrieval.
- Output: ES, NES, nominal p-value, **GSEApy FDR q-value**, semicolon-separated leading-edge genes, provenance and timing.
- Fail closed on execution errors and invalid result schemas; no alternative engine or statistical fallback.
- Engine-specific GSEA FDR is **not** relabeled as BH adjusted p-values.
- Permutation resolution, tie handling, and GSEApy vs R fgsea numerical differences require independent scientific review.

## Verification

1. `python -m py_compile omicsgpt/gsea_execution.py`
2. `python -m pytest -q tests/test_gsea_execution.py`
3. `python -m pytest -q tests/`
4. Run a Docker integration against a **TEST_ONLY** GMT using saved 75-gene DE results. This is an engineering test, **not** pathway discovery.
5. Only after the engine passes, obtain a versioned real Reactome GMT and full-transcriptome DE results for biological validation and independent R comparison.

Do not commit until local tests and real GSEApy integration succeed.

### Condensed-data Docker smoke test

From the repository root, after extracting the ZIP:

```powershell
docker run --rm `
  --workdir /app `
  --mount "type=bind,source=$($PWD.Path),target=/app" `
  --entrypoint python `
  oncology-agent:v1 `
  -m scripts.check_gsea_condensed `
  --results /app/benchmark_outputs/v2_2/differential_expression.csv `
  --outdir /app/benchmark_outputs/v2_6b2_test_only
```

The GMT is deliberately constructed from top/bottom ranked genes and labeled `TEST_ONLY`. It checks integration, not unbiased enrichment inference; its significance values must **not** be interpreted biologically. Never substitute it for a real versioned Reactome resource.
