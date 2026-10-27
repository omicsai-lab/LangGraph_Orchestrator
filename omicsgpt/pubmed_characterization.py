"""Dynamic, conservative PubMed study characterization (no LLM or clinical inference)."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence

from omicsgpt.evidence_schema import EvidenceRecord
from omicsgpt.study_characterization import (
    StudyCharacterization, StudyCharacterizationError, characterize_pubmed_record,
    summarize_characterizations,
)


@dataclass(frozen=True)
class PublicationContext:
    """Original source text and claim-specific, unreviewed study scaffold."""
    characterization: StudyCharacterization
    title: str
    abstract: str | None
    publication_date_raw: str | None
    source_fingerprint: str

    def to_dict(self) -> dict:
        return {
            'characterization': self.characterization.to_dict(),
            'title': self.title,
            'abstract': self.abstract,
            'publication_date_raw': self.publication_date_raw,
            'source_fingerprint': self.source_fingerprint,
        }


def prepare_publication_contexts(
    records: Sequence[EvidenceRecord], *, claim: str, disease: str
) -> tuple[list[PublicationContext], dict]:
    """Preserve each retrieved paper; do not guess design, directness, or direction.

    Reject invalid or duplicate records rather than silently dropping them.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise StudyCharacterizationError('Expected a nonempty sequence of PubMed records')
    if not isinstance(claim, str) or not claim.strip() or claim != claim.strip():
        raise StudyCharacterizationError('claim must be a nonblank trimmed string')
    if not isinstance(disease, str) or not disease.strip() or disease != disease.strip():
        raise StudyCharacterizationError('disease must be a nonblank trimmed string')
    output: list[PublicationContext] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, EvidenceRecord) or record.source != 'PubMed' or record.kind != 'literature' or record.status != 'found':
            raise StudyCharacterizationError('All records must be found PubMed literature evidence')
        identity = (record.source_record_id, record.subject_id)
        if identity in seen:
            raise StudyCharacterizationError(f'Duplicate publication/gene: {identity}')
        seen.add(identity)
        title = record.attributes.get('title')
        abstract = record.attributes.get('abstract')
        date = record.attributes.get('publication_date_raw')
        if not isinstance(title, str) or not title.strip():
            raise StudyCharacterizationError('PubMed title missing')
        if abstract is not None and (not isinstance(abstract, str) or not abstract.strip()):
            raise StudyCharacterizationError('Invalid abstract')
        if date is not None and not isinstance(date, str):
            raise StudyCharacterizationError('Invalid publication date')
        scaffold = characterize_pubmed_record(record, claim=claim, disease=disease)
        output.append(PublicationContext(
            characterization=scaffold, title=title, abstract=abstract,
            publication_date_raw=date, source_fingerprint=record.fingerprint(),
        ))
    summary = summarize_characterizations([item.characterization for item in output])
    audit = {
        'n_publications': len(output),
        'pmids': [item.characterization.pmid for item in output],
        'claim': claim, 'disease': disease,
        'abstracts_available': sum(item.abstract is not None for item in output),
        'extraction_method': 'deterministic_scaffold_no_study_inference',
        'all_study_findings_unassessed': True,
        'summary': summary,
    }
    return output, audit


def export_contexts_json(contexts: Sequence[PublicationContext], audit: dict) -> str:
    """Portable JSON export without changing the source records."""
    if not isinstance(contexts, (list, tuple)) or any(not isinstance(c, PublicationContext) for c in contexts):
        raise TypeError('Expected PublicationContext objects')
    return json.dumps({'audit': audit, 'publications': [c.to_dict() for c in contexts]},
                      indent=2, ensure_ascii=False, allow_nan=False)
