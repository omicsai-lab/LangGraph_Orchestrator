import pytest
from omicsgpt.claim_literature import claim_focus_terms, extract_explicit_descriptors, run_claim_literature
from omicsgpt.pubmed_adapter import PubMedPolicy, search_pubmed
from omicsgpt.evidence_schema import EvidenceRecord
from omicsgpt.pubmed_characterization import prepare_publication_contexts


def record(abstract):
    return EvidenceRecord(source='PubMed', source_record_id='123', kind='literature', status='found',
        subject_id='BRCA1', subject_id_type='gene_symbol', organism='Homo sapiens',
        retrieved_at_utc='2026-10-27T00:00:00+00:00', finding='Publication retrieved: Test',
        attributes={'title':'Test', 'abstract':abstract, 'publication_date_raw':None})


def context(abstract):
    return prepare_publication_contexts([record(abstract)], claim='BRCA1 expression predicts treatment response', disease='breast cancer')[0][0]


def test_focus_terms():
    assert claim_focus_terms('BRCA1 expression predicts treatment response') == ('expression','predict','treatment','response')


def test_focus_unknown_abstains():
    assert claim_focus_terms('BRCA1 is important') == ()

@pytest.mark.parametrize('claim', ['', ' bad ', None])
def test_bad_claim(claim):
    with pytest.raises(ValueError): claim_focus_terms(claim)


def test_literal_animal_and_design():
    result = extract_explicit_descriptors(context('A cohort study in mice examined Brca1.'))
    assert result.study_system == 'animal'
    assert result.study_design == 'cohort'
    assert result.system_excerpt.lower() == 'mice'
    assert result.finding_relation == 'not_assessed'


def test_mixed_system_abstains_on_single_label():
    result = extract_explicit_descriptors(context('Mouse models and patients were evaluated.'))
    assert result.study_system == 'mixed'
    assert result.system_excerpt is None


def test_unknown_does_not_guess():
    result = extract_explicit_descriptors(context('BRCA1 expression was evaluated.'))
    assert result.study_system == 'not_reported'
    assert result.study_design == 'not_reported'
    assert result.claim_directness == 'unclear'


def test_no_abstract():
    result = extract_explicit_descriptors(context(None))
    assert result.study_system == 'not_reported'


def test_claim_search_passes_audited_focus():
    captured = {}
    def fake(gene, disease, *, policy, focus_terms):
        captured['terms'] = focus_terms
        return [record('Patients were included in a cohort study.')], {'status':'found','query':'recorded'}
    items, audit = run_claim_literature('BRCA1','breast cancer','BRCA1 expression predicts treatment response',search_fn=fake)
    assert captured['terms'] == ('expression','predict','treatment','response')
    assert audit['n_extracted'] == 1
    assert items[0]['automated_descriptors']['study_system'] == 'human'
    assert items[0]['automated_descriptors']['finding_relation'] == 'not_assessed'


def test_failure_keeps_audit():
    def fake(*args, **kwargs): return [], {'status':'unavailable'}
    items, audit = run_claim_literature('BRCA1','breast cancer','BRCA1 expression',search_fn=fake)
    assert items == []
    assert audit['retrieval']['status'] == 'unavailable'


def test_pubmed_query_escapes_focus():
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'esearchresult':{'idlist':[],'count':'0'}}
    class Session:
        def get(self, url, *, params, timeout):
            assert '"expression"[Title/Abstract]' in params['term']
            assert '"BRCA1"[Title/Abstract]' in params['term']
            return Response()
    records, audit = search_pubmed('BRCA1','breast cancer',focus_terms=('expression',),session=Session())
    assert audit['status'] == 'not_found'
    assert audit['focus_terms'] == ['expression']

@pytest.mark.parametrize('terms',[('bad\nterm',),('a',)*6, ('x','x'), ('',)])
def test_invalid_focus_terms(terms):
    with pytest.raises(ValueError): search_pubmed('BRCA1','breast cancer',focus_terms=terms)
