import pandas as pd
import pytest
from scripts.audit_full_counts import summarize_library_sizes


def test_library_sums_preserve_input():
    x = pd.DataFrame({'g1':[1,3], 'g2':[2,4]}, index=['s1','s2'])
    original = x.copy(deep=True)
    table, summary = summarize_library_sizes(x)
    assert table['library_size'].tolist() == [3,7]
    assert summary['median'] == 5
    pd.testing.assert_frame_equal(x, original)


def test_zero_library_rejected():
    x = pd.DataFrame({'g':[0,2]}, index=['s1','s2'])
    with pytest.raises(ValueError, match='zero total'):
        summarize_library_sizes(x)


def test_fractional_values_rejected():
    x = pd.DataFrame({'g':[1.5,2.0]}, index=['s1','s2'])
    with pytest.raises(ValueError, match='integer'):
        summarize_library_sizes(x)


def test_duplicate_ids_rejected():
    x = pd.DataFrame({'g':[1,2]}, index=['s1','s1'])
    with pytest.raises(ValueError, match='unique'):
        summarize_library_sizes(x)
