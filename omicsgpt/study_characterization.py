"""Validated, claim-specific study characterization; no automatic inference or scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Literal

from omicsgpt.evidence_schema import EvidenceRecord

StudySystem = Literal['human', 'animal', 'cell_line', 'organoid', 'computational', 'mixed', 'other', 'not_reported']
StudyDesign = Literal['randomized_trial', 'nonrandomized_intervention', 'cohort', 'case_control', 'cross_sectional', 'case_report', 'experimental', 'systematic_review', 'narrative_review', 'other', 'not_reported']
Directness = Literal['direct', 'indirect_network', 'unrelated', 'unclear']
FindingRelation = Literal['supports', 'contradicts', 'mixed', 'inconclusive', 'not_assessed']
InformationLevel = Literal['metadata_only', 'abstract', 'full_text']
Verification = Literal['unreviewed', 'researcher_reviewed']

class StudyCharacterizationError(ValueError):
    pass


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StudyCharacterizationError(f'{name} must be a nonblank trimmed string')


@dataclass(frozen=True)
class StudyCharacterization:
    pmid: str
    claim: str
    gene: str
    disease: str
    study_system: StudySystem = 'not_reported'
    study_design: StudyDesign = 'not_reported'
    directness: Directness = 'unclear'
    finding_relation: FindingRelation = 'not_assessed'
    information_level: InformationLevel = 'metadata_only'
    verification: Verification = 'unreviewed'
    outcome: str | None = None
    population_or_model: str | None = None
    limitations: tuple[str, ...] = ()
    supporting_excerpt: str | None = None
    extraction_method: Literal['manual', 'automated_unverified'] = 'manual'

    def __post_init__(self):
        if not isinstance(self.pmid, str) or not re.fullmatch(r'[0-9]+', self.pmid):
            raise StudyCharacterizationError('PMID must be a numeric string')
        for key in ('claim', 'gene', 'disease'):
            _nonempty(getattr(self, key), key)
        choices = {
            'study_system': StudySystem, 'study_design': StudyDesign,
            'directness': Directness, 'finding_relation': FindingRelation,
            'information_level': InformationLevel, 'verification': Verification,
            'extraction_method': Literal['manual', 'automated_unverified'],
        }
        for key, typ in choices.items():
            if getattr(self, key) not in typ.__args__:
                raise StudyCharacterizationError(f'Invalid {key}')
        for key in ('outcome', 'population_or_model', 'supporting_excerpt'):
            value = getattr(self, key)
            if value is not None:
                _nonempty(value, key)
        if not isinstance(self.limitations, tuple) or any(not isinstance(x, str) or not x.strip() for x in self.limitations):
            raise StudyCharacterizationError('Invalid limitations')
        if self.information_level == 'metadata_only' and self.finding_relation != 'not_assessed':
            raise StudyCharacterizationError('Metadata alone cannot establish finding direction')
        if self.finding_relation in ('supports', 'contradicts', 'mixed') and not self.supporting_excerpt:
            raise StudyCharacterizationError('Directional finding requires an attributed supporting excerpt')
        if self.verification == 'researcher_reviewed' and self.extraction_method == 'automated_unverified':
            raise StudyCharacterizationError('Automated extraction cannot be labeled researcher-reviewed')
        json.dumps(asdict(self), allow_nan=False)

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def characterize_pubmed_record(record: EvidenceRecord, *, claim: str, disease: str) -> StudyCharacterization:
    """Produce an explicit unassessed scaffold; never infer study type or findings."""
    if not isinstance(record, EvidenceRecord) or record.source != 'PubMed' or record.kind != 'literature' or record.status != 'found':
        raise StudyCharacterizationError('Requires a found PubMed literature EvidenceRecord')
    _nonempty(record.subject_id, 'gene')
    return StudyCharacterization(pmid=record.source_record_id, claim=claim, gene=record.subject_id, disease=disease,
                                 information_level='abstract' if record.attributes.get('abstract') else 'metadata_only')


def summarize_characterizations(studies: list[StudyCharacterization]) -> dict:
    """Descriptive counts only; never pool heterogeneous evidence into a consensus."""
    if not isinstance(studies, list) or any(not isinstance(s, StudyCharacterization) for s in studies):
        raise StudyCharacterizationError('Expected list of StudyCharacterization')
    identities = [(s.pmid, s.claim, s.gene, s.disease) for s in studies]
    if len(identities) != len(set(identities)):
        raise StudyCharacterizationError('Duplicate PMID/claim/gene/disease appraisal')
    from collections import Counter
    return {'n_studies': len(studies),
            'by_system': dict(sorted(Counter(s.study_system for s in studies).items())),
            'by_directness': dict(sorted(Counter(s.directness for s in studies).items())),
            'by_finding_relation': dict(sorted(Counter(s.finding_relation for s in studies).items())),
            'n_researcher_reviewed': sum(s.verification == 'researcher_reviewed' for s in studies),
            'note': 'Descriptive characterization only; no pooled effect, quality score, or consensus inferred.'}
