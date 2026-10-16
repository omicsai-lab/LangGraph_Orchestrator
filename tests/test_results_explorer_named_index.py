"""Regression tests for DE result tables exported with a named gene index."""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from omicsgpt.results_explorer import SelectionPolicy, select_candidates, prepare_volcano


@pytest.mark.parametrize('index_name', [None, 'gene_id', 'symbol'])
@pytest.mark.parametrize('strategy', ['balanced', 'significance', 'effect'])
def test_selection_accepts_named_gene_index_without_mutating_source(index_name, strategy):
    df = pd.DataFrame({
        'log2FoldChange': [2.0, -2.0, 1.0, -1.0],
        'pvalue': [0.001, 0.002, 0.003, 0.004],
        'padj': [0.01, 0.02, 0.03, 0.04],
    }, index=pd.Index(['G1', 'G2', 'G3', 'G4'], name=index_name))
    original = df.copy(deep=True)
    policy = SelectionPolicy(strategy=strategy, n=2)
    first, audit = select_candidates(df, policy)
    second, audit2 = select_candidates(df, policy)
    assert first == second
    assert audit == audit2
    assert len(first) == len(set(first))
    assert set(first).issubset(set(df.index))
    assert_frame_equal(df, original, check_exact=True)


def test_volcano_accepts_named_gene_index_without_mutating_source():
    df = pd.DataFrame({
        'log2FoldChange': [1.5, -0.8],
        'pvalue': [0.01, 0.1],
        'padj': [0.03, 0.2],
    }, index=pd.Index(['A', 'B'], name='gene_id'))
    original = df.copy(deep=True)
    plot, audit = prepare_volcano(df)
    assert list(plot.index) == ['A', 'B']
    assert audit['n_total'] == 2
    assert_frame_equal(df, original, check_exact=True)


def test_duplicate_gene_identifiers_rejected():
    df = pd.DataFrame({'log2FoldChange': [1.0, -1.0], 'pvalue': [.01, .02], 'padj': [.02, .03]}, index=pd.Index(['A', 'A'], name='gene_id'))
    with pytest.raises(ValueError, match='unique gene IDs'):
        select_candidates(df, SelectionPolicy())
