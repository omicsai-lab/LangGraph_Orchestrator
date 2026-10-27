import json
import pytest
from omicsgpt.pubmed_adapter import PubMedPolicy, search_pubmed

XML = b'''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>BRCA1 and breast cancer</ArticleTitle><Abstract><AbstractText Label="BACKGROUND">First sentence.</AbstractText><AbstractText Label="RESULTS">Second sentence.</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'''

class Response:
    def __init__(self, *, data=None, content=b'', status=200):
        self.data, self.content, self.status = data, content, status
    def raise_for_status(self):
        import requests
        if self.status >= 400:
            raise requests.exceptions.HTTPError(str(self.status))
    def json(self):
        return self.data

class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)

def clock(): return '2026-10-27T20:00:00+00:00'

def test_found_preserves_both_abstract_sections_and_provenance():
    s = Session([Response(data={'esearchresult': {'idlist': ['123'], 'count': '1'}}), Response(content=XML)])
    records, audit = search_pubmed('BRCA1', 'breast cancer', session=s, clock=clock)
    assert len(records) == 1 and records[0].status == 'found'
    assert records[0].source_record_id == '123'
    assert 'BACKGROUND: First sentence.' in records[0].attributes['abstract']
    assert 'RESULTS: Second sentence.' in records[0].attributes['abstract']
    assert records[0].direction == 'context_only'
    assert records[0].underlying_study_ids == ('PMID:123',)
    assert audit['returned_count'] == 1
    assert len(s.calls) == 2
    assert s.calls[0][1]['params']['retmax'] == 20

def test_empty_search_is_not_found():
    s = Session([Response(data={'esearchresult': {'idlist': [], 'count': '0'}})])
    records, audit = search_pubmed('BRCA1', 'breast cancer', session=s, clock=clock)
    assert records[0].status == audit['status'] == 'not_found'
    assert len(s.calls) == 1

def test_http_failure_is_unavailable():
    records, audit = search_pubmed('BRCA1', 'breast cancer', session=Session([Response(status=403)]), clock=clock)
    assert records[0].status == audit['status'] == 'unavailable'

def test_missing_requested_pmid_is_error():
    records, audit = search_pubmed('BRCA1', 'breast cancer', session=Session([
        Response(data={'esearchresult': {'idlist': ['123', '456'], 'count': '2'}}), Response(content=XML)]), clock=clock)
    assert records[0].status == audit['status'] == 'error'

def test_malformed_xml_is_error():
    records, audit = search_pubmed('BRCA1', 'breast cancer', session=Session([
        Response(data={'esearchresult': {'idlist': ['123'], 'count': '1'}}), Response(content=b'<broken>')]), clock=clock)
    assert records[0].status == audit['status'] == 'error'

def test_deep_dive_limit():
    assert PubMedPolicy(mode='deep_dive').limit == 100
    assert PubMedPolicy().limit == 20
    with pytest.raises(ValueError): PubMedPolicy(mode='fast', max_records=21)
    with pytest.raises(ValueError): PubMedPolicy(mode='deep_dive', max_records=101)

@pytest.mark.parametrize('gene,disease', [('', 'breast cancer'), ('BRCA1', ''), (' BRCA1', 'breast cancer')])
def test_invalid_queries(gene, disease):
    with pytest.raises(ValueError): search_pubmed(gene, disease, session=Session([]), clock=clock)

def test_unexpected_pmid_is_error():
    bad = XML.replace(b'<PMID>123</PMID>', b'<PMID>999</PMID>')
    records, _ = search_pubmed('BRCA1', 'breast cancer', session=Session([
        Response(data={'esearchresult': {'idlist': ['123'], 'count': '1'}}), Response(content=bad)]), clock=clock)
    assert records[0].status == 'error'

def test_truncation_is_explicit():
    s = Session([Response(data={'esearchresult': {'idlist': ['123'], 'count': '500'}}), Response(content=XML)])
    _, audit = search_pubmed('BRCA1', 'breast cancer', session=s, clock=clock)
    assert audit['truncated_by_limit'] is True
