import numpy as np
import pandas as pd
import pytest
from omicsgpt.results_explorer import SelectionPolicy, select_candidates, prepare_volcano


@pytest.fixture
def de():
    return pd.DataFrame({'log2FoldChange': [2, 3, -4, -1, 0.2, np.nan],
                         'pvalue': [0, 0.001, 0.003, 0.02, 0.8, np.nan],
                         'padj': [0.01, 0.01, 0.02, 0.04, 0.9, np.nan]},
                        index=['B', 'A', 'C', 'D', 'E', 'F'])


def test_balanced(de):
    ids, audit = select_candidates(de, SelectionPolicy(n=1))
    assert ids == ['A', 'C'] and audit['n_selected'] == 2


def test_effect(de):
    assert select_candidates(de, SelectionPolicy(strategy='effect', n=2))[0] == ['C', 'A']


def test_significance_tie_by_gene(de):
    assert select_candidates(de, SelectionPolicy(strategy='significance', n=2))[0] == ['A', 'B']


def test_manual_does_not_filter(de):
    assert select_candidates(de, SelectionPolicy(strategy='manual', manual_genes=('E', 'B', 'E')))[0] == ['E', 'B']


def test_manual_unknown_fails(de):
    with pytest.raises(ValueError, match='Unknown'):
        select_candidates(de, SelectionPolicy(strategy='manual', manual_genes=('BAD',)))


def test_fdr_eligibility(de):
    assert select_candidates(de, SelectionPolicy(strategy='effect', n=8, fdr=.01))[0] == ['A', 'B']


def test_threshold(de):
    assert select_candidates(de, SelectionPolicy(strategy='effect', n=8, min_abs_log2fc=3))[0] == ['C', 'A']


def test_zero_p_display_only(de):
    before = de.copy(deep=True)
    table, audit = prepare_volcano(de)
    assert audit['n_zero_p_capped'] == 1
    assert table.loc['B', 'pvalue'] == 0
    assert np.isfinite(table.loc['B', 'volcano_y'])
    pd.testing.assert_frame_equal(de, before)


def test_missing_excluded(de):
    table, audit = prepare_volcano(de)
    assert not table.loc['F', 'plot_eligible']
    assert audit['n_unplottable'] == 1


def test_adjusted_axis(de):
    table, audit = prepare_volcano(de, p_column='padj')
    assert audit['n_zero_p_capped'] == 0
    assert table.loc['A', 'volcano_y'] == pytest.approx(2.0)


@pytest.mark.parametrize('kwargs', [{'strategy':'unknown'}, {'n':0}, {'fdr':float('nan')}, {'min_abs_log2fc':-1}])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        SelectionPolicy(**kwargs)


def test_nonfinite_input(de):
    de.loc['A', 'log2FoldChange'] = np.inf
    with pytest.raises(ValueError, match='nonfinite'):
        prepare_volcano(de)


def test_bad_p(de):
    de.loc['A', 'padj'] = 2
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        select_candidates(de, SelectionPolicy())


def test_deterministic_order_independent(de):
    p = SelectionPolicy(strategy='effect', n=3)
    assert select_candidates(de, p)[0] == select_candidates(de.iloc[::-1], p)[0]
