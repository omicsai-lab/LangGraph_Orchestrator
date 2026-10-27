# OmicsGPT V2.7A.3d — Claim-aware PubMed retrieval and conservative descriptors

## Scope

- `search_pubmed` now accepts optional `focus_terms` (at most five literal terms). These are ANDed with the original gene/disease query as an OR group. Original calls behave unchanged.
- `claim_focus_terms` deterministically extracts **only explicitly listed** vocabulary from the supplied claim. The exact query and focus terms are recorded; no LLM query rewriting occurs.
- `run_claim_literature` returns individual source-linked publication contexts and narrow descriptors of study system/design when their exact cues appear in an abstract.
- Mixed study systems are flagged; study designs with multiple cues are left unreported.
- Claim directness, findings, evidence strength, and risk of bias **are never inferred**. They remain unassessed even when a paper is retrieved.
- This is a heuristic first pass, **not** validated scientific study extraction or comprehensive literature searching. Query vocabulary may miss relevant studies, and keyword cues may be misleading.

## Tests and next scientific validation

Use the focused tests, then full regression suite, then a bounded live test. Inspect returned PMIDs and queries. Do not use the extracted descriptors to automatically rank or discard evidence. Next evaluate source-grounded LLM extraction and claim relevance against reviewed examples before making substantive biological claims.

## Dependencies

Existing `requests`, `EvidenceRecord`, `PubMedPolicy`, `StudyCharacterization`, and `PublicationContext` are reused. No new third-party dependency or Streamlit integration.
