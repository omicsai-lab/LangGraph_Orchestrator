# V2.7A.3c — Dynamic publication characterization

This module connects the existing PubMed retrieval adapter to the existing claim-specific study characterization contract. It does **not** use a fixed publication list, LLM, or manual workbook.

## Contract
- Input: found PubMed `EvidenceRecord` objects, an explicit claim, and disease.
- Output: per-publication context (PMID, title, full retrieved abstract when available, publication date, original record fingerprint, conservative `StudyCharacterization`) plus batch audit.
- All design, model system, directness, finding direction, and appraisal fields remain unknown/unassessed until supported extraction is implemented and validated.
- Never silently drop failed API statuses or duplicate PMIDs; handle upstream `audit['status']` before calling the module.
- No evidence pooling, literature scoring, or scientific consensus.
- No network calls or new dependencies.

## Validation
`python -m pytest -q tests/test_pubmed_characterization.py`

Use `scripts/check_pubmed_characterization.py` for one optional live request with a small cap. It saves results under a caller-selected benchmark output directory; no source-code changes or Git-tracked results.

## Next checkpoint
Implement a bounded, source-grounded extraction method with explicit abstention and a small **dynamic** evaluation protocol. Manual expert review is useful for independent spot checks, but is not a runtime dependency or a fixed list of allowed publications.
