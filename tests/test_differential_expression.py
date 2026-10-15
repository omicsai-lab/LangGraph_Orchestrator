import numpy as np
import pandas as pd
import pytest
from omicsgpt.differential_expression import FilterPolicy, filter_counts, run_pydeseq2


def example():
    counts = pd.DataFrame({'keep': [10, 11, 0, 10, 11], 'drop': [0, 0, 0, 0, 0]},
                          index=[f'S{i}' for i in range(5)])
    return counts


def test_filter_audit_and_source_preserved():
    original = example()
    snapshot = original.copy(deep=True)
    filtered, audit = filter_counts(original, FilterPolicy(10, 0.4))
    assert filtered.columns.tolist() == ['keep']
    assert audit['min_samples'] == 2
    assert audit['excluded_genes'] == ['drop']
    pd.testing.assert_frame_equal(original, snapshot)


@pytest.mark.parametrize('bad', [float('nan'), -1, 1.5, float('inf')])
def test_invalid_counts_fail_closed(bad):
    counts = example()
    counts.loc['S0', 'keep'] = bad
    with pytest.raises(ValueError):
        filter_counts(counts, FilterPolicy())


def test_no_genes_survive():
    with pytest.raises(ValueError, match='No genes'):
        filter_counts(example(), FilterPolicy(100, 0.5))


def test_fraction_uses_ceiling():
    _, audit = filter_counts(example(), FilterPolicy(10, 0.21))
    assert audit['min_samples'] == 2


def test_unaligned_metadata_rejected_before_engine_import():
    counts = example()
    metadata = pd.DataFrame({'condition': ['A','A','B','B','B']}, index=list(reversed(counts.index)))
    with pytest.raises(ValueError, match='sample order'):
        run_pydeseq2(counts, metadata, group_col='condition', numerator='B', reference='A')
