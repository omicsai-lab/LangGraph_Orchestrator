"""Red-first contract tests. These will fail until the intake module is implemented.

Dependencies for this test module: pytest, pandas, numpy.
Expected future API: omicsgpt.intake.inspect_bulk_counts(counts, metadata, sample_id_col)
Return object: .orientation, .status, .issues; must not mutate inputs.
"""
import numpy as np
import pandas as pd
import pytest

from omicsgpt.intake import inspect_bulk_counts


@pytest.fixture
def sample_data():
    counts = pd.DataFrame({"S1": [10, 0], "S2": [20, 5], "S3": [30, 7], "S4": [40, 9]}, index=["G1", "G2"])
    metadata = pd.DataFrame({"sample_id": ["S1", "S2", "S3", "S4"], "group": ["A", "A", "B", "B"]})
    return counts, metadata


def test_detect_genes_by_samples(sample_data):
    counts, metadata = sample_data
    result = inspect_bulk_counts(counts, metadata, sample_id_col="sample_id")
    assert result.orientation == "genes_by_samples"
    assert result.status == "ready_for_confirmation"


def test_detect_samples_by_genes(sample_data):
    counts, metadata = sample_data
    result = inspect_bulk_counts(counts.T, metadata, sample_id_col="sample_id")
    assert result.orientation == "samples_by_genes"


@pytest.mark.parametrize("bad_value", [np.nan, -1, 1.5, np.inf, "bad"])
def test_invalid_count_blocks_analysis(sample_data, bad_value):
    counts, metadata = sample_data
    counts = counts.astype(object)
    counts.loc["G1", "S1"] = bad_value
    result = inspect_bulk_counts(counts, metadata, sample_id_col="sample_id")
    assert result.status == "blocked"
    assert result.issues


def test_duplicate_sample_id_blocks_analysis(sample_data):
    counts, metadata = sample_data
    metadata.loc[1, "sample_id"] = "S1"
    result = inspect_bulk_counts(counts, metadata, sample_id_col="sample_id")
    assert result.status == "blocked"


def test_no_mutation_of_source(sample_data):
    counts, metadata = sample_data
    original_counts = counts.copy(deep=True)
    original_metadata = metadata.copy(deep=True)
    inspect_bulk_counts(counts, metadata, sample_id_col="sample_id")
    pd.testing.assert_frame_equal(counts, original_counts)
    pd.testing.assert_frame_equal(metadata, original_metadata)


def test_unmatched_sample_ids_need_resolution(sample_data):
    counts, metadata = sample_data
    metadata.loc[3, "sample_id"] = "NOT_IN_COUNTS"
    result = inspect_bulk_counts(counts, metadata, sample_id_col="sample_id")
    assert result.status in {"needs_user_resolution", "blocked"}
