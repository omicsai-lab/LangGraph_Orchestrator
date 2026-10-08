import pandas as pd
import pytest
from omicsgpt.intake import inspect_bulk_counts, prepare_confirmed_counts


def fixture_data():
    counts = pd.DataFrame({'S1': [0, 3], 'S2': [0, 4]}, index=['G1', 'G2'])
    meta = pd.DataFrame({'sample_id': ['S2', 'S1'], 'condition': ['case', 'control']})
    return counts, meta


def test_confirmed_preparation_transposes_and_reorders_without_mutation():
    counts, meta = fixture_data()
    original = counts.copy(deep=True)
    result = prepare_confirmed_counts(counts, meta, 'sample_id', 'genes_by_samples')
    assert list(result.index) == ['S2', 'S1']
    assert list(result.columns) == ['G1', 'G2']
    assert result.loc['S2', 'G2'] == 4
    pd.testing.assert_frame_equal(counts, original)


def test_orientation_confirmation_required():
    counts, meta = fixture_data()
    with pytest.raises(ValueError, match='Confirmed orientation'):
        prepare_confirmed_counts(counts, meta, 'sample_id', 'samples_by_genes')


def test_all_zero_genes_are_reported_not_removed():
    counts, meta = fixture_data()
    report = inspect_bulk_counts(counts, meta, 'sample_id')
    assert report.all_zero_genes == 1
    assert report.status == 'ready_for_confirmation'


def test_extra_count_samples_require_resolution():
    counts, meta = fixture_data()
    counts['S3'] = [0, 1]
    report = inspect_bulk_counts(counts, meta, 'sample_id')
    assert report.status == 'needs_user_resolution'


def test_duplicate_gene_ids_block():
    counts, meta = fixture_data()
    counts.index = ['G1', 'G1']
    assert inspect_bulk_counts(counts, meta, 'sample_id').status == 'blocked'


def test_both_axes_matching_is_ambiguous():
    counts = pd.DataFrame([[1, 2], [3, 4]], index=['S1', 'S2'], columns=['S1', 'S2'])
    meta = pd.DataFrame({'sample_id': ['S1', 'S2']})
    assert inspect_bulk_counts(counts, meta, 'sample_id').status == 'needs_user_resolution'
