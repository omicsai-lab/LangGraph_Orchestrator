import pytest
from omicsgpt.mygene_adapter import lookup_human_gene

T = '2026-10-27T12:00:00+00:00'

class Response:
    def __init__(self, data, status_code=200):
        self.data, self.status_code = data, status_code
    def json(self):
        return self.data

class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

def run(data, status=200, symbol='BRCA1'):
    return lookup_human_gene(symbol, session=Session(Response(data, status)), retrieved_at_utc=T)

HIT = {'_id':'672','symbol':'BRCA1','taxid':9606,'name':'BRCA1 DNA repair associated','alias':['RNF53']}

def test_exact_match_and_provenance():
    record = run({'hits':[HIT], 'total':1})
    assert record.status == 'found'
    assert record.source_record_id == '672'
    assert record.direction == 'context_only'
    assert record.attributes['aliases'] == ['RNF53']
    assert len(record.source_payload_sha256) == 64

def test_no_match():
    assert run({'hits':[], 'total':0}).status == 'not_found'

def test_alias_is_not_identity():
    assert run({'hits':[dict(HIT, symbol='OTHER')], 'total':1}).status == 'not_found'

def test_wrong_species():
    assert run({'hits':[dict(HIT, taxid=10090)], 'total':1}).status == 'not_found'

def test_ambiguous_exact_matches():
    assert run({'hits':[HIT,dict(HIT, _id='999')], 'total':2}).status == 'error'

def test_truncated_results():
    assert run({'hits':[HIT], 'total':101}).status == 'error'

@pytest.mark.parametrize('code,expected',[(429,'unavailable'),(503,'unavailable'),(403,'error')])
def test_http_errors(code, expected):
    assert run({}, code).status == expected

def test_invalid_response():
    assert run({'unexpected':True}).status == 'error'

def test_malformed_annotation():
    assert run({'hits':[dict(HIT, alias={'bad':'shape'})], 'total':1}).status == 'error'

def test_empty_stable_id():
    assert run({'hits':[dict(HIT, _id='')], 'total':1}).status == 'error'

def test_input_validation():
    with pytest.raises(ValueError): lookup_human_gene('BRCA1 OR *')
    with pytest.raises(ValueError): lookup_human_gene('BRCA1', timeout=0)

def test_request_params():
    session = Session(Response({'hits':[HIT], 'total':1}))
    lookup_human_gene('BRCA1', session=session, retrieved_at_utc=T)
    url, opts = session.calls[0]
    assert url.startswith('https://mygene.info/')
    assert opts['params']['species'] == 'human'
    assert opts['params']['q'] == 'symbol:BRCA1'
