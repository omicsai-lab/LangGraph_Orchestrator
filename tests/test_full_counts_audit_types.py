"""Regression tests for the library-size audit's strict count representations."""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scripts.audit_full_counts import summarize_library_sizes


def matrix(values):
    return pd.DataFrame(values, index=['S1', 'S2'], columns=['G1', 'G2'])


def test_decimal_integer_strings_match_integer_dtype():
    numeric = matrix([[2, 3], [0, 7]])
    text = numeric.astype(str)
    original = text.copy(deep=True)
    left, summary_a = summarize_library_sizes(numeric)
    right, summary_b = summarize_library_sizes(text)
    assert_frame_equal(left, right)
    assert summary_a == summary_b
    assert left['library_size'].tolist() == [5, 7]
    assert_frame_equal(text, original)


@pytest.mark.parametrize('bad', ['1.5', '1.0', '-1', '1e2', '', ' ', 'nan', 'inf',
                                       None, float('nan'), float('inf'), 1.0, True,
                                       str(2**63)])
def test_invalid_count_values_rejected(bad):
    values = matrix([['1', '2'], ['3', '4']])
    values.loc['S1', 'G1'] = bad
    with pytest.raises(ValueError):
        summarize_library_sizes(values)


def test_zero_total_rejected():
    with pytest.raises(ValueError, match='zero total'):
        summarize_library_sizes(matrix([['0', '0'], ['1', '2']]))


def test_duplicate_gene_identifiers_rejected():
    values = matrix([['1', '2'], ['3', '4']])
    values.columns = ['G1', 'G1']
    with pytest.raises(ValueError, match='unique'):
        summarize_library_sizes(values)


def test_leading_zeros_are_exact_integer_values():
    tab, _ = summarize_library_sizes(matrix([['001', '002'], ['003', '004']]))
    assert tab['library_size'].tolist() == [3, 7]
