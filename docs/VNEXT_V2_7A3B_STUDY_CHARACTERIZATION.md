# V2.7A.3b — Claim-specific literature characterization contract

## Scope
This checkpoint adds a **manual/researcher-review-ready schema**, not a literature classifier, quality scorer, or systematic-review engine. It does not make network calls or modify PubMed retrieval, GSEA, PyDESeq2, or Streamlit.

## Contract
- Each record is specific to `(PMID, gene, disease, claim)`. A paper can inform multiple claims; these must be characterized separately.
- Record study system, study design, directness, finding relationship, information level, review state, population/model, outcome, limitations, and supporting excerpt when applicable.
- Defaults are deliberately `not_reported`, `unclear`, `not_assessed`, and `unreviewed`.
- `metadata_only` cannot support directional findings; directional findings require an attributed excerpt and at least abstract-level information.
- A machine-extracted record must not be represented as researcher-reviewed without an explicit subsequent review step.
- Aggregation is **descriptive counts only**, not pooled evidence or consensus.

## Existing integration
`characterize_pubmed_record(record, claim=..., disease=...)` accepts the `EvidenceRecord` from V2.7A.3 and returns an **unassessed scaffold**. It does not infer study design from titles or abstracts.

## Scientific limitations
A supporting excerpt is a traceability aid, **not** independent verification of the claim, quality appraisal, or protection against extraction errors. The schema does not yet check that excerpts occur verbatim in source documents. Abstract-only evidence cannot establish detailed risk of bias. Real-PMID examples are identifiers for integration checks, not manually verified study classifications.

## Next validation
Create a small, researcher-reviewed set of publications with prespecified claims, annotated study system/design and excerpts. Compare future AI extraction against these labels; preserve disagreements and missingness. Do not auto-synthesize evidence strength before this evaluation.

## Tests
`python -m pytest -q tests/test_study_characterization.py`

## Dependencies
Python standard library and existing `omicsgpt.evidence_schema`; no new third-party dependencies.
