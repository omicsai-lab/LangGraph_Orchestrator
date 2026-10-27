"""Source-independent, auditable evidence records for OmicsGPT.

No network, AI inference, or automatic evidence scoring occurs here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Literal

EvidenceStatus = Literal['found', 'not_found', 'unavailable', 'error', 'not_applicable']
EvidenceKind = Literal['gene_annotation', 'pathway', 'literature', 'interaction', 'tissue_expression', 'protein', 'disease_target', 'clinical', 'structure', 'other']
EvidenceDirection = Literal['supporting', 'contradictory', 'mixed', 'context_only', 'undetermined']
EvidenceBasis = Literal['primary_experiment', 'curated_annotation', 'computational_prediction', 'aggregated_evidence', 'literature_metadata', 'other']

class EvidenceValidationError(ValueError):
    """Invalid evidence record; do not silently repair."""


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise EvidenceValidationError(f'{label} must be a nonblank trimmed string')
    return value


@dataclass(frozen=True)
class EvidenceRecord:
    source: str
    source_record_id: str
    kind: EvidenceKind
    status: EvidenceStatus
    subject_id: str
    subject_id_type: str
    organism: str
    retrieved_at_utc: str
    source_version: str | None = None
    source_url: str | None = None
    disease_context: str | None = None
    tissue_context: str | None = None
    direction: EvidenceDirection = 'undetermined'
    basis: EvidenceBasis = 'other'
    finding: str | None = None
    limitations: tuple[str, ...] = ()
    underlying_study_ids: tuple[str, ...] = ()
    source_payload_sha256: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ('source', 'source_record_id', 'subject_id', 'subject_id_type', 'organism'):
            _required(getattr(self, key), key)
        if self.kind not in EvidenceKind.__args__ or self.status not in EvidenceStatus.__args__:
            raise EvidenceValidationError('Invalid evidence kind or status')
        if self.direction not in EvidenceDirection.__args__ or self.basis not in EvidenceBasis.__args__:
            raise EvidenceValidationError('Invalid direction or basis')
        try:
            parsed = datetime.fromisoformat(self.retrieved_at_utc.replace('Z', '+00:00'))
        except (ValueError, AttributeError) as exc:
            raise EvidenceValidationError('Invalid retrieval timestamp') from exc
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise EvidenceValidationError('retrieved_at_utc must be UTC and timezone-aware')
        if self.status == 'found' and not self.finding:
            raise EvidenceValidationError('Found evidence must include a finding')
        if self.status != 'found' and self.finding is not None:
            raise EvidenceValidationError('Non-found statuses cannot contain findings')
        if self.status != 'found' and self.direction != 'undetermined':
            raise EvidenceValidationError('Non-found statuses cannot imply a direction')
        if self.source_url is not None and not self.source_url.startswith(('https://', 'http://')):
            raise EvidenceValidationError('Source URL must be HTTP(S)')
        if self.source_payload_sha256 is not None and (len(self.source_payload_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.source_payload_sha256)):
            raise EvidenceValidationError('Invalid SHA-256 digest')
        if not isinstance(self.limitations, tuple) or not all(isinstance(v, str) and v.strip() for v in self.limitations):
            raise EvidenceValidationError('Limitations must be a tuple of nonempty strings')
        if not isinstance(self.underlying_study_ids, tuple) or not all(isinstance(v, str) and v.strip() for v in self.underlying_study_ids):
            raise EvidenceValidationError('Study IDs must be a tuple of nonempty strings')
        if not isinstance(self.attributes, dict):
            raise EvidenceValidationError('Attributes must be a dictionary')
        try:
            json.dumps(asdict(self), allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise EvidenceValidationError('Record contains non-JSON or nonfinite values') from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return sha256(canonical.encode('utf-8')).hexdigest()


def make_status_record(*, source: str, source_record_id: str, kind: EvidenceKind,
                       status: EvidenceStatus, subject_id: str, subject_id_type: str,
                       organism: str, retrieved_at_utc: str, limitations: tuple[str, ...] = ()) -> EvidenceRecord:
    """Explicit absence/failure record; never infer negative biological evidence."""
    if status == 'found':
        raise EvidenceValidationError('Use EvidenceRecord for found evidence')
    return EvidenceRecord(source=source, source_record_id=source_record_id, kind=kind,
                          status=status, subject_id=subject_id, subject_id_type=subject_id_type,
                          organism=organism, retrieved_at_utc=retrieved_at_utc, limitations=limitations)
