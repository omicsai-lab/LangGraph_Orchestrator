"""Non-fitting GSEA input contract tests."""
import hashlib
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from omicsgpt.gsea_contract import GSEAContractError, prepare_gsea_input, read_gmt, validate_ranking


def fixture():
    return pd.DataFrame({'stat': [1.0, -3.0, 2.0, 0.5], 'padj': [0.1, 0.2, 0.3, 0.4]}, index=pd.Index(['G1','G2','G3','G4'], name='gene_id'))


def test_rank_sort_ties_and_source_unchanged():
    d = fixture()
    d.loc['G1', 'stat'] = 2.0
    before = d.copy(deep=True)
    rank = validate_ranking(d)
    assert rank.index.tolist() == ['G1','G3','G4','G2']
    assert_frame_equal(d, before)


@pytest.mark.parametrize('alter', [
    lambda d: d.rename(index={'G2':'G1'}),
    lambda d: d.rename(index={'G2':''}),
    lambda d: d.assign(stat=['1','2','3','4']),
    lambda d: d.assign(stat=[1, float('nan'), 2, 3]),
    lambda d: d.assign(stat=[1, float('inf'), 2, 3]),
    lambda d: d.assign(stat=[1, 1, 1, 1]),
    lambda d: d.drop(columns='stat'),
])
def test_invalid_ranking_fails_closed(alter):
    with pytest.raises(GSEAContractError):
        validate_ranking(alter(fixture()))


def test_gmt_checksum_and_membership():
    raw = b'P1\tdescription\tG1\tG2\nP2\tother\tG3\tG4\n'
    sets, audit = read_gmt(raw)
    assert sets['P1'] == ('G1', 'G2')
    assert audit['gmt_sha256'] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('raw', [b'', b'P1\td\n', b'P1\td\tG1\tG1\n', b'P1\td\tG1\nP1\td\tG2\n', b'P1\td\tG1\n\n', b'P1\td\tG1\t\n'])
def test_invalid_gmt_fails_closed(raw):
    with pytest.raises(GSEAContractError):
        read_gmt(raw)


def test_coverage_filter_and_no_mutation():
    d = fixture()
    before = d.copy(deep=True)
    raw = b'P1\td\tG1\tG2\tG9\nP2\td\tG3\nP3\td\tG99\n'
    obj = prepare_gsea_input(d, raw, min_size=2, max_size=3, resource_release='test-release')
    assert obj.gene_sets == {'P1': ('G1','G2')}
    assert obj.audit['genes_matching_any_pathway'] == 3
    assert obj.audit['pathways_retained'] == 1
    assert len(obj.audit['pathways_excluded']) == 2
    assert_frame_equal(d, before)


def test_requires_release_and_nonempty_pathway_overlap():
    with pytest.raises(GSEAContractError):
        prepare_gsea_input(fixture(), b'P\td\tG1\n', min_size=1)
    with pytest.raises(GSEAContractError):
        prepare_gsea_input(fixture(), b'P\td\tUNKNOWN\n', min_size=1, resource_release='test')


def test_size_parameter_validation():
    with pytest.raises(GSEAContractError):
        prepare_gsea_input(fixture(), b'P\td\tG1\n', min_size=3, max_size=2, resource_release='test')
