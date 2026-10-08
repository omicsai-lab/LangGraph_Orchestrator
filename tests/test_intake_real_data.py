"""Real-data regression checks for the existing bulk RNA intake module.

Run from repository root: python -m pytest -q tests/test_intake_real_data.py
This checks pandas-loaded data, NOT strict CSV parsing or Streamlit integration.
"""
from pathlib import Path
import pandas as pd
from omicsgpt.intake import inspect_bulk_counts, prepare_confirmed_counts

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / 'bc_counts_transposed_condensed.csv'
METADATA = ROOT / 'bc_meta.csv'


def test_condensed_breast_cancer_intake_against_independent_reference():
    assert COUNTS.is_file(), f'Missing test fixture: {COUNTS}'
    assert METADATA.is_file(), f'Missing test fixture: {METADATA}'
    counts = pd.read_csv(COUNTS, index_col=0)
    metadata = pd.read_csv(METADATA)
    source_counts = counts.copy(deep=True)
    source_metadata = metadata.copy(deep=True)

    # Reference expectations independent of the intake implementation.
    assert counts.shape == (187, 110)
    assert metadata.shape == (187, 2)
    assert metadata['Sample'].is_unique
    assert counts.index.is_unique and counts.columns.is_unique
    assert set(counts.index) == set(metadata['Sample'])
    assert len(set(counts.columns).intersection(metadata['Sample'])) == 0
    assert counts.notna().all().all()
    assert (counts.to_numpy() >= 0).all()
    assert ((counts.to_numpy() % 1) == 0).all()
    expected_zero_genes = int((counts.sum(axis=0) == 0).sum())
    assert expected_zero_genes == 3

    report = inspect_bulk_counts(counts, metadata, sample_id_col='Sample')
    assert report.status == 'ready_for_confirmation'
    assert report.orientation == 'samples_by_genes'
    assert report.n_metadata_samples == 187
    assert report.matched_row_ids == 187
    assert report.matched_column_ids == 0
    assert report.all_zero_genes == expected_zero_genes
    assert not report.issues

    prepared = prepare_confirmed_counts(counts, metadata, 'Sample', 'samples_by_genes')
    assert prepared.shape == (187, 110)
    pd.testing.assert_frame_equal(prepared, counts.loc[metadata['Sample'].tolist()])
    pd.testing.assert_frame_equal(counts, source_counts)
    pd.testing.assert_frame_equal(metadata, source_metadata)


def test_real_data_transposed_orientation_produces_identical_prepared_matrix():
    counts = pd.read_csv(COUNTS, index_col=0)
    metadata = pd.read_csv(METADATA)
    transposed = counts.T.copy(deep=True)
    original_transposed = transposed.copy(deep=True)
    report = inspect_bulk_counts(transposed, metadata, 'Sample')
    assert report.status == 'ready_for_confirmation'
    assert report.orientation == 'genes_by_samples'
    assert report.matched_row_ids == 0
    assert report.matched_column_ids == 187
    prepared = prepare_confirmed_counts(transposed, metadata, 'Sample', 'genes_by_samples')
    pd.testing.assert_frame_equal(prepared, counts.loc[metadata['Sample'].tolist()])
    pd.testing.assert_frame_equal(transposed, original_transposed)
