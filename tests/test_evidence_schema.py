import dataclasses
import math
import pytest
from omicsgpt.evidence_schema import EvidenceRecord, EvidenceValidationError, make_status_record

BASE = dict(source='PubMed', source_record_id='PMID:123', kind='literature', status='found',
            subject_id='BRCA1', subject_id_type='HGNC_symbol', organism='Homo sapiens',
            retrieved_at_utc='2026-10-27T12:00:00Z', finding='Study reports an association.',
            basis='primary_experiment', direction='supporting')

def test_record_is_frozen():
    r = EvidenceRecord(**BASE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.source = 'Other'

def test_fingerprint_deterministic():
    assert EvidenceRecord(**BASE).fingerprint() == EvidenceRecord(**BASE).fingerprint()

def test_dict_serializable():
    assert EvidenceRecord(**BASE).to_dict()['source'] == 'PubMed'

@pytest.mark.parametrize('changes', [
    {'source': ''}, {'kind': 'fake'}, {'status': 'fake'}, {'direction': 'fake'},
    {'retrieved_at_utc': '2026-10-27T12:00:00'}, {'retrieved_at_utc': 'bad'},
    {'source_url': 'file:///private'}, {'source_payload_sha256': 'abc'},
    {'attributes': {'bad': math.nan}}, {'limitations': ['not tuple']},
    {'finding': None}, {'status': 'not_found'},
])
def test_invalid(changes):
    with pytest.raises(EvidenceValidationError):
        EvidenceRecord(**(BASE | changes))

def test_absence_not_negative_evidence():
    r = make_status_record(source='HPA', source_record_id='query:BRCA1', kind='tissue_expression',
        status='unavailable', subject_id='BRCA1', subject_id_type='HGNC_symbol',
        organism='Homo sapiens', retrieved_at_utc='2026-10-27T12:00:00Z')
    assert r.finding is None and r.direction == 'undetermined'

def test_status_helper_rejects_found():
    with pytest.raises(EvidenceValidationError):
        make_status_record(source='HPA', source_record_id='q', kind='protein', status='found',
            subject_id='BRCA1', subject_id_type='symbol', organism='Homo sapiens',
            retrieved_at_utc='2026-10-27T12:00:00Z')

def test_study_provenance_not_deduplicated_automatically():
    r = EvidenceRecord(**(BASE | {'underlying_study_ids': ('PMID:123',)}))
    assert r.underlying_study_ids == ('PMID:123',)
