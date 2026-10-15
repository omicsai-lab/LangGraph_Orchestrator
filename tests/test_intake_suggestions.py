import pandas as pd
from omicsgpt.intake_suggestions import suggest_columns, compact_group_sizes


def fixture():
    counts = pd.DataFrame({'G1': [1, 2, 3, 4], 'G2': [4, 3, 2, 1]}, index=['S1','S2','S3','S4'])
    meta = pd.DataFrame({'condition': ['high','low','high','low'], 'Sample': ['S1','S2','S3','S4']})
    return counts, meta


def test_sample_matching_not_first_column():
    counts, meta = fixture()
    s = suggest_columns(counts, meta)
    assert (s.sample_id_column, s.grouping_column, s.sample_id_match_count) == ('Sample', 'condition', 4)


def test_transposed_counts():
    counts, meta = fixture()
    assert suggest_columns(counts.T, meta).sample_id_column == 'Sample'


def test_advisory_when_no_matching_ids():
    counts, meta = fixture()
    meta['Sample'] = ['X1', 'X2', 'X3', 'X4']
    s = suggest_columns(counts, meta)
    assert s.sample_id_match_count == 0
    assert 'verify' in s.sample_id_reason


def test_no_grouping_when_all_identifiers():
    counts, meta = fixture()
    meta['condition'] = ['A','B','C','D']
    assert suggest_columns(counts, meta).grouping_column is None


def test_compact_groups():
    rows, remaining = compact_group_sizes({f'G{i}': 1 for i in range(30)}, limit=5)
    assert len(rows) == 5 and remaining == 25


def test_source_unchanged():
    counts, meta = fixture()
    before_counts, before_meta = counts.copy(deep=True), meta.copy(deep=True)
    suggest_columns(counts, meta)
    pd.testing.assert_frame_equal(counts, before_counts)
    pd.testing.assert_frame_equal(meta, before_meta)
