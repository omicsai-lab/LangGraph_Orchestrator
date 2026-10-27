# OmicsGPT V2.7A.2 — MyGene.info adapter

- Source: MyGene.info v3 query endpoint; gene-symbol lookup restricted to human (taxid 9606).
- Requires exact, case-sensitive `symbol` equality. Aliases and top search hits are never silently substituted.
- Truncated or multiple exact results are errors requiring review; absence is `not_found`, not negative biological evidence.
- Network failures, HTTP errors, and invalid response formats are explicitly reported as evidence statuses.
- Found records use the committed `EvidenceRecord` schema, preserve selected annotations and aliases, and record source ID, URL, timestamp and canonical response-record SHA-256.
- Annotation evidence is `context_only`, not proof of disease association, therapeutic actionability, or protein expression.
- No Streamlit, LangGraph, API keys, or LLM calls are required.
- Tests use deterministic offline response fixtures. A separate small live smoke test should precede production enablement.
- Known limitations: query results are not a complete gene identifier reconciliation workflow; source version is not inferred from retrieval time; upstream evidence remains aggregated.
