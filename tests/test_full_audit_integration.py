"""Full audit-path regression tests using a small realistic CSV fixture."""
import json
import sys
from pathlib import Path
from subprocess import run

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from scripts.audit_full_counts import _validated_count_array, summarize_library_sizes


def _fixture(tmp_path):
    counts = tmp_path / 'counts.csv'
    metadata = tmp_path / 'metadata.csv'
    # Unnamed row-ID column; sample-by-gene orientation; real benchmark group names.
    counts.write_text(',G1,G2,G3\nS1,12,0,1\nS2,15,1,0\nS3,0,20,1\nS4,0,30,0\n', encoding='utf-8')
    metadata.write_text('Sample,condition\nS1,risk category: High\nS2,risk category: High\nS3,risk category: Average\nS4,risk category: Average\n', encoding='utf-8')
    return counts, metadata


def _run(tmp_path, *, min_count=10, min_fraction=0.5, counts_override=None):
    counts, metadata = _fixture(tmp_path)
    if counts_override is not None:
        counts.write_text(counts_override, encoding='utf-8')
    out = tmp_path / 'output'
    result = run([sys.executable, '-m', 'scripts.audit_full_counts',
                  '--counts', str(counts), '--metadata', str(metadata),
                  '--outdir', str(out), '--min-count', str(min_count),
                  '--min-fraction', str(min_fraction)],
                 capture_output=True, text=True, timeout=30)
    return result, out


def test_end_to_end_unnamed_id_and_integer_strings(tmp_path):
    proc, out = _run(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    audit = json.loads((out / 'intake_audit.json').read_text())
    assert audit['samples'] == 4
    assert audit['genes_input'] == 3
    assert audit['genes_all_zero'] == 0
    assert audit['filter']['genes_retained'] == 2
    assert audit['filter']['genes_excluded'] == 1
    assert pd.read_csv(out / 'sample_library_sizes.csv')['library_size'].tolist() == [13, 16, 21, 30]
    assert {'inspect_seconds','prepare_seconds','numeric_validation_seconds',
            'library_summary_seconds','filter_summary_seconds'} <= set(audit['stage_timings_seconds'])


def test_threshold_zero_is_explicitly_supported(tmp_path):
    proc, out = _run(tmp_path, min_count=0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads((out / 'intake_audit.json').read_text())['filter']['genes_retained'] == 3


@pytest.mark.parametrize('bad', ['-1','1.5','nan',''])
def test_bad_counts_fail_without_outputs(tmp_path, bad):
    source = f',G1,G2,G3\nS1,{bad},0,1\nS2,15,1,0\nS3,0,20,1\nS4,0,30,0\n'
    proc, out = _run(tmp_path, counts_override=source)
    assert proc.returncode != 0
    assert not (out / 'intake_audit.json').exists()
    assert not (out / 'sample_library_sizes.csv').exists()


def test_numeric_matrix_matches_string_matrix_without_mutation():
    text = pd.DataFrame({'G1':['12','0'], 'G2':['3','20']}, index=['S1','S2'])
    original = text.copy(deep=True)
    numeric = _validated_count_array(text)
    table, _ = summarize_library_sizes(text, validated_values=numeric)
    assert table['library_size'].tolist() == [15, 20]
    assert np.count_nonzero(numeric >= 10, axis=0).tolist() == [1, 1]
    assert_frame_equal(text, original)


def test_rejects_mismatched_validated_array_shape():
    text = pd.DataFrame({'G1':['1','2']}, index=['S1','S2'])
    with pytest.raises(ValueError, match='incompatible'):
        summarize_library_sizes(text, validated_values=np.ones((1, 2), dtype=np.int64))
