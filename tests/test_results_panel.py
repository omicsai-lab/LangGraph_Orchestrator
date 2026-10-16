"""Presentation contract tests; no Streamlit or statistical engine required."""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from omicsgpt.results_panel import (explorer_payload, STRATEGY_EXPLANATIONS,
    format_probability, display_results_table, candidate_marker_style)


def fixture():
    return pd.DataFrame({'baseMean':[10,20,30,40],
        'log2FoldChange':[3.0,-2.0,0.2,1.0],
        'pvalue':[0.0001,0.002,0.6,0.03],
        'padj':[0.001,0.02,0.8,0.04]},
        index=pd.Index(['A','B','C','D'],name='gene_id'))


@pytest.mark.parametrize('strategy,n,expected',[
    ('balanced',1,['A','B']),('significance',2,['A','B']),
    ('effect',2,['A','B']),('manual',2,['D','A'])])
def test_selection_and_plot(strategy,n,expected):
    original=fixture()
    before=original.copy(deep=True)
    kwargs={'manual_genes':('D','A')} if strategy=='manual' else {}
    output=explorer_payload(original,strategy=strategy,n=n,**kwargs)
    assert output['selected_genes']==expected
    assert len(output['volcano'])==4
    assert set(output['volcano'].loc[output['volcano']['candidate_selected'],'gene_id'])==set(expected)
    assert_frame_equal(original,before)


def test_display_threshold_changes_only_selection():
    data=fixture()
    a=explorer_payload(data,strategy='effect',n=4,min_abs_log2fc=0)
    b=explorer_payload(data,strategy='effect',n=4,min_abs_log2fc=2.5)
    assert len(a['selected_genes'])>len(b['selected_genes'])
    assert len(a['volcano'])==len(b['volcano'])


def test_named_index_is_not_ambiguous():
    data=fixture()
    assert explorer_payload(data)['volcano']['gene_id'].tolist()==list(data.index)


def test_bad_selection_rejected():
    with pytest.raises(ValueError):
        explorer_payload(fixture(),strategy='not-supported')


def test_missing_p_values_are_audited():
    data=fixture()
    data.loc['C','pvalue']=float('nan')
    result=explorer_payload(data)
    assert result['volcano_audit']['n_unplottable']==1


def test_zero_p_is_plot_only():
    data=fixture()
    data.loc['A','pvalue']=0
    result=explorer_payload(data)
    assert result['volcano_audit']['n_zero_p_capped']==1
    assert data.loc['A','pvalue']==0


@pytest.mark.parametrize('strategy', ['balanced', 'significance', 'effect', 'manual'])
def test_strategy_descriptions_present(strategy):
    label, explanation = STRATEGY_EXPLANATIONS[strategy]
    assert label and len(explanation) > 50

@pytest.mark.parametrize('value,expected', [
    (0.2536, '0.2536'), (0.01234, '0.0123'),
    (0.0000345, '3.45e-05'), (2.8e-13, '2.80e-13'),
    (0.0, '0 (computational)'), (float('nan'), 'NA'),
])
def test_probability_formatting(value, expected):
    assert format_probability(value) == expected


def test_display_table_preserves_original_numeric_values():
    data = fixture()
    before = data.copy(deep=True)
    shown = display_results_table(data)
    assert shown.loc[0, 'pvalue'] == '1.00e-04'
    assert shown.loc[0, 'padj'] == '0.0010'
    assert shown.loc[0, 'gene_id'] == 'A'
    assert_frame_equal(data, before)


def test_candidate_marker_visible_and_distinct():
    marker = candidate_marker_style()
    assert marker['symbol'] == 'star'
    assert marker['color'] != '#111111'
    assert marker['size'] >= 12
    assert marker['line']['width'] > 0
