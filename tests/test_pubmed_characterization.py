from datetime import datetime, timezone
import json
import pytest

from omicsgpt.evidence_schema import EvidenceRecord, make_status_record
from omicsgpt.pubmed_characterization import prepare_publication_contexts, export_contexts_json
from omicsgpt.study_characterization import StudyCharacterizationError

STAMP = '2026-10-27T12:00:00+00:00'


def paper(pmid='12345', abstract='METHODS: A cell model was evaluated.', gene='BRCA1'):
    return EvidenceRecord(source='PubMed', source_record_id=pmid, kind='literature', status='found',
        subject_id=gene, subject_id_type='gene_symbol', organism='Homo sapiens',
        retrieved_at_utc=STAMP, direction='context_only', basis='literature_metadata',
        finding='Publication retrieved: Example', attributes={'title': 'Example',
        'abstract': abstract, 'publication_date_raw': '2024'})


def test_dynamic_batch_and_source_preservation():
    records = [paper(), paper('67890', None)]
    before = [r.fingerprint() for r in records]
    items, audit = prepare_publication_contexts(records, claim='BRCA1 predicts response', disease='breast cancer')
    assert audit['n_publications'] == 2 and audit['abstracts_available'] == 1
    assert audit['pmids'] == ['12345', '67890']
    assert items[0].characterization.information_level == 'abstract'
    assert items[1].characterization.information_level == 'metadata_only'
    assert all(i.characterization.finding_relation == 'not_assessed' for i in items)
    assert all(i.characterization.study_design == 'not_reported' for i in items)
    assert [r.fingerprint() for r in records] == before
    assert json.loads(export_contexts_json(items, audit))['publications'][0]['source_fingerprint'] == before[0]


def test_no_unsupported_inference_from_abstract():
    items, _ = prepare_publication_contexts([paper(abstract='Randomized trial proves efficacy.')],
        claim='Treatment improves survival', disease='breast cancer')
    c = items[0].characterization
    assert c.study_design == 'not_reported'
    assert c.finding_relation == 'not_assessed'
    assert c.directness == 'unclear'


def test_duplicate_rejected():
    with pytest.raises(StudyCharacterizationError, match='Duplicate'):
        prepare_publication_contexts([paper(), paper()], claim='Claim', disease='Disease')


@pytest.mark.parametrize('bad', [[], [None], [paper(), None]])
def test_bad_input_rejected(bad):
    with pytest.raises(StudyCharacterizationError):
        prepare_publication_contexts(bad, claim='Claim', disease='Disease')


def test_failure_status_is_not_evidence():
    failed = make_status_record(source='PubMed', source_record_id='query:123', kind='literature',
        status='unavailable', subject_id='BRCA1', subject_id_type='gene_symbol',
        organism='Homo sapiens', retrieved_at_utc=STAMP)
    with pytest.raises(StudyCharacterizationError):
        prepare_publication_contexts([failed], claim='Claim', disease='Disease')


def test_invalid_title_rejected():
    r = paper()
    from dataclasses import replace
    with pytest.raises(StudyCharacterizationError, match='title'):
        prepare_publication_contexts([replace(r, attributes={'abstract': 'a'})], claim='Claim', disease='Disease')


def test_invalid_claim_rejected():
    with pytest.raises(StudyCharacterizationError):
        prepare_publication_contexts([paper()], claim=' ', disease='Disease')
