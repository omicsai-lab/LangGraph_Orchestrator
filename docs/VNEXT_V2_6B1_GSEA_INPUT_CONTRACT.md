# OmicsGPT V2.6B.1 — Preranked GSEA input contract

This checkpoint adds **input validation only**. No GSEA, statistical fitting, API calls, or gene-identifier conversion occurs.

- Ranking is based on the existing signed `stat` column, ordered descending with gene-ID tie-breaking. Missing, nonfinite, nonnumeric, or constant rankings fail closed.
- GMT is a local UTF-8, tab-separated gene-set resource. The file's SHA-256 digest and declared release are recorded; malformed records and duplicate pathway or gene-membership identifiers fail closed.
- Gene identifiers are matched **exactly and case-sensitively**. No symbol correction, version stripping, or cross-species mapping occurs. Coverage and gene-set exclusions are audited.
- Minimum/maximum gene-set sizes apply to the overlap with the ranked universe, not to the original GMT gene-set size.
- The returned ranking is a new Series; source DataFrames are not modified. Gene-set outputs and audit are for downstream consumption; callers should not mutate them.
- The caller must obtain a versioned Reactome human gene-symbol GMT separately, confirm its license and origin, and supply its declared release. **No bundled GMT is represented as a genuine Reactome release.**
- No biological inference is justified from the 75-gene condensed development benchmark. Validate on a broad ranked gene universe after full-transcriptome DE is available.
- This checkpoint does not silently remove genes with missing statistics; a future explicit exclusion policy can be designed and audited if scientifically justified.

## Local checks

```powershell
python -m py_compile omicsgpt/gsea_contract.py
python -m pytest -q tests/test_gsea_contract.py
python -m pytest -q tests/
git status --short
```

## Next

V2.6B.2 will integrate an established GSEA engine (proposed: GSEApy), independently test it against synthetic fixtures and an R reference, and explicitly record random seed, permutations, engine version, and results provenance. No Streamlit modifications are made here.
