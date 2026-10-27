from dataclasses import replace
import pytest
from omicsgpt.evidence_schema import EvidenceRecord
from omicsgpt.study_characterization import (
    StudyCharacterization, StudyCharacterizationError, characterize_pubmed_record,
    summarize_characterizations,
)


def record(abstract=None):
    return EvidenceRecord(source='PubMed', source_record_id='29687286', kind='literature', status='found',
        subject_id='BRCA1', subject_id_type='gene_symbol', organism='Homo sapiens',
        retrieved_at_utc='2026-10-27T00:00:00Z', finding='Publication retrieved',
        attributes={'abstract': abstract})


def base():
    return StudyCharacterization(pmid='29687286', claim='BRCA1 expression predicts response to therapy',
                                 gene='BRCA1', disease='breast cancer')


def test_scaffold_is_unassessed():
    s = characterize_pubmed_record(record('An abstract.'), claim='BRCA1 is prognostic', disease='breast cancer')
    assert s.information_level == 'abstract' and s.study_system == 'not_reported'
    assert s.finding_relation == 'not_assessed' and s.verification == 'unreviewed'


def test_no_abstract_is_metadata_only():
    assert characterize_pubmed_record(record(), claim='claim', disease='breast cancer').information_level == 'metadata_only'

@pytest.mark.parametrize('field,value', [('pmid','abc'),('claim',''),('gene',' BRCA1'),('disease',''),
    ('study_system','clinical'),('study_design','rct'),('directness','yes'),('finding_relation','positive'),
    ('information_level','pdf'),('verification','verified'),('extraction_method','AI')])
def test_invalid_fields(field, value):
    with pytest.raises(StudyCharacterizationError):
        replace(base(), **{field: value})


def test_direction_needs_excerpt():
    with pytest.raises(StudyCharacterizationError):
        replace(base(), finding_relation='supports')


def test_metadata_cannot_claim_direction():
    with pytest.raises(StudyCharacterizationError):
        replace(base(), finding_relation='supports', supporting_excerpt='Some text')


def test_abstract_direction_with_excerpt():
    s = replace(base(), information_level='abstract', finding_relation='supports', supporting_excerpt='Observed association')
    assert s.finding_relation == 'supports'


def test_automated_cannot_claim_reviewed():
    with pytest.raises(StudyCharacterizationError):
        replace(base(), extraction_method='automated_unverified', verification='researcher_reviewed')


def test_summary_separates_study_types():
    a = replace(base(), study_system='human', study_design='cohort')
    b = replace(base(), pmid='39528827', study_system='animal', study_design='experimental')
    result = summarize_characterizations([a, b])
    assert result['by_system'] == {'animal': 1, 'human': 1}
    assert result['n_researcher_reviewed'] == 0


def test_duplicate_claim_appraisal_rejected():
    with pytest.raises(StudyCharacterizationError):
        summarize_characterizations([base(), base()])


def test_same_paper_distinct_claim_allowed():
    assert summarize_characterizations([base(), replace(base(), claim='BRCA1 is involved in DNA repair')])['n_studies'] == 2


def test_deterministic_fingerprint():
    assert base().fingerprint() == base().fingerprint()


def test_source_not_found_rejected():
    r = replace(record(), status='not_found', finding=None)
    with pytest.raises(StudyCharacterizationError):
        characterize_pubmed_record(r, claim='claim', disease='breast cancer')


def test_no_automatic_study_design_from_title():
    r = replace(record(), attributes={'title':'Randomized clinical trial', 'abstract':'Patients were enrolled.'})
    assert characterize_pubmed_record(r, claim='claim', disease='breast cancer').study_design == 'not_reported'
