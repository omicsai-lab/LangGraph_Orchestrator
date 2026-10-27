# V2.7A.3 — PubMed evidence retrieval

## Scope
- Reuses NCBI E-Utilities (`esearch` and `efetch`) from the legacy OmicsGPT architecture.
- Returns source-independent `EvidenceRecord` instances, plus an explicit search audit.
- No LLM calls, embeddings, automated paper rejection, study-quality scores, or clinical claims.
- Fast mode retrieves up to 20 records; Deep Dive retrieves up to 100 in one bounded search. Deep Dive is a broader first pass, **not** an exhaustive systematic review.
- Results are labeled `context_only`: a keyword match is not evidence of biological causality or even direct gene relevance.

## Provenance and safeguards
- Query, mode, requested limit, total matches, truncation, PMID order, retrieval timestamp, source URLs, and payload checksum are recorded.
- Abstract sections are retained when available; missing abstracts are not treated as failed searches.
- Search and fetch errors produce explicit `unavailable` or `error` records; zero hits produce `not_found`.
- Failed/partial fetches do not silently yield a successful partial publication list.
- No patient-level data should be submitted to PubMed. Queries contain only gene and disease terms.

## Limitations and next steps
- Search currently uses literal gene and disease terms in title/abstract; no alias expansion, MeSH normalization, study-design classification, pagination, or citation deduplication beyond PMID validation.
- The default relevance sort is provider-defined and may change over time.
- Deep Dive should later add transparent query expansion and reviewable source screening, not merely higher `retmax`.
- External API access, provider rate limits, caching, and endpoint behavior require a live smoke test.
- Use recorded-response tests for routine CI; live API tests should be opt-in.

## Install and validate
```
python -m py_compile omicsgpt/pubmed_adapter.py
python -m pytest -q tests/test_pubmed_adapter.py
python -m pytest -q tests/
```

No changes to statistical modules or Streamlit. Do not commit before local validation.
