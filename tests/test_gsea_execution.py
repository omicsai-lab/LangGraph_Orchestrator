"""GSEA executor contract tests, runnable without GSEApy or network access."""
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from omicsgpt.gsea_contract import prepare_gsea_input, GSEAContractError
from omicsgpt.gsea_execution import GSEASettings, GSEAExecutionError, run_preranked_gsea


def fixture():
    names = [f'G{i:02d}' for i in range(30)]
    frame = pd.DataFrame({'stat': list(range(15, 0, -1)) + list(range(-1, -16, -1))}, index=names)
    gmt = ('TOP\tTEST_ONLY\t' + '\t'.join(names[:10]) + '\nBOTTOM\tTEST_ONLY\t' + '\t'.join(names[-10:]) + '\n').encode()
    return prepare_gsea_input(frame, gmt, min_size=5, resource_release='synthetic-test-1')

class FakeFit:
    def __init__(self, frame): self.res2d = frame

def fake_engine(**kwargs):
    assert kwargs['outdir'] is None
    return FakeFit(pd.DataFrame({'Term':['TOP','BOTTOM'], 'ES':[0.8,-0.7], 'NES':[1.8,-1.5],
       'NOM p-val':[0.01,0.02], 'FDR q-val':[0.03,0.04], 'Lead_genes':['G00;G01','G20;G21']}))


def test_output_and_provenance():
    data = fixture(); before = data.ranking.copy(deep=True)
    result = run_preranked_gsea(data, engine=fake_engine)
    assert result['results']['pathway'].tolist() == ['TOP','BOTTOM']
    assert result['manifest']['ranked_genes'] == 30
    assert result['manifest']['seed'] == 42
    assert result['manifest']['input_audit']['resource_release'] == 'synthetic-test-1'
    assert_series_equal(data.ranking, before)

@pytest.mark.parametrize('kw', [{'permutation_num':0},{'seed':-1},{'threads':0},{'weight':float('nan')},{'weight':-1}])
def test_invalid_settings(kw):
    with pytest.raises(ValueError): GSEASettings(**kw)

def test_engine_failure_is_not_silenced():
    def broken(**kwargs): raise RuntimeError('test failure')
    with pytest.raises(GSEAExecutionError, match='test failure'):
        run_preranked_gsea(fixture(), engine=broken)

def test_bad_schema_is_rejected():
    with pytest.raises(GSEAExecutionError, match='schema'):
        run_preranked_gsea(fixture(), engine=lambda **k: FakeFit(pd.DataFrame({'Term':['TOP']})))

def test_unknown_leading_edge_is_rejected():
    def broken(**kwargs):
        f = fake_engine(**kwargs)
        f.res2d.loc[0,'Lead_genes'] = 'NOT_A_GENE'
        return f
    with pytest.raises(GSEAExecutionError, match='Leading-edge'):
        run_preranked_gsea(fixture(), engine=broken)

def test_unsorted_rank_is_rejected():
    data = fixture(); data.ranking.iloc[0] = -999
    with pytest.raises(GSEAContractError): run_preranked_gsea(data, engine=fake_engine)

def test_settings_forwarded():
    seen = {}
    def engine(**kwargs): seen.update(kwargs); return fake_engine(**kwargs)
    run_preranked_gsea(fixture(), settings=GSEASettings(permutation_num=250,seed=9,threads=2,weight=0.5), engine=engine)
    assert (seen['permutation_num'],seen['seed'],seen['threads'],seen['weight']) == (250,9,2,0.5)
