# OmicsGPT V2.7A.1 — Source-independent evidence records

## Scope
Schema only. No live API calls, LLM scoring, GSEA, DE, or Streamlit changes.

## Design
- Evidence records separate **status** (`found`, `not_found`, `unavailable`, `error`, `not_applicable`) from biological **direction** (`supporting`, `contradictory`, `mixed`, `context_only`, `undetermined`).
- A non-found response cannot assert a finding or direction. Missing evidence is not negative evidence.
- Record subject identifier and identifier type, organism, source and record ID, retrieval timestamp, optional source version/URL/payload hash, evidence basis, disease/tissue context, limitations, and underlying study IDs.
- No universal evidence score, silent source deduplication, inferred source quality, or automatic gene mapping.
- `attributes` holds JSON-serializable provider-specific fields; adapters must validate their semantics separately.
- Fingerprints are SHA-256 hashes of canonical JSON **including retrieval timestamp**. They identify exact records, not equivalent biological findings.
- `frozen=True` prevents field reassignment but nested dictionaries are mutable; adapters should treat them as read-only and make defensive copies at boundaries.

## Intended adapters
First: MyGene.info and PubMed. Then STRING, Open Targets, HPA and UniProt. Conditional: OncoKB, ClinicalTrials.gov. Optional: AlphaFold. Reactome gene-set acquisition remains a separate, nonblocking workstream.

## Validation
Run `python -m pytest -q tests/test_evidence_schema.py` then `python -m pytest -q tests/`. Do not commit until local suite passes.
