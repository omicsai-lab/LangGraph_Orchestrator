import json
from pathlib import Path
import pandas as pd
import pytest
from omicsgpt.csv_import import read_strict_csv, CSVImportError
from omicsgpt.design import inspect_design
from omicsgpt.intake import inspect_bulk_counts
from omicsgpt.intake_review import make_intake_review

ROOT = Path(__file__).resolve().parents[1]


def test_preserve_leading_zeros_and_hash():
    raw = b'id,S01,S02\n001,1,2\n002,0,5\n'
    parsed = read_strict_csv(raw, use_identifier_as_index=True)
    assert parsed.frame.index.tolist() == ['001', '002']
    assert parsed.frame.loc['001', 'S01'] == '1'
    assert len(parsed.sha256) == 64
    assert parsed.n_rows == 2

@pytest.mark.parametrize('raw', [
    b'id,A,A\nx,1,2\n', b'id,A\nx,1\nx,2\n',
    b'id,A\nx,1,2\n', b'id,A\nx,1\ny\n',
    b'id,A\n,1\n', b'id,A\nx,1\n\n',
])
def test_reject_bad_csv(raw):
    with pytest.raises(CSVImportError):
        read_strict_csv(raw)


def test_counts_invalid_after_strict_import():
    counts = read_strict_csv(b'id,S1,S2\nG1,3,4.5\n', use_identifier_as_index=True).frame
    meta = pd.DataFrame({'Sample':['S1','S2']})
    assert inspect_bulk_counts(counts, meta, 'Sample').status == 'blocked'


def test_explicit_contrast_and_metadata_order():
    metadata = pd.DataFrame({'condition':['High','Average','High','Average'], 'batch':['A','A','B','B']})
    report = inspect_design(metadata, group_col='condition', numerator='High', reference='Average', covariates=('batch',))
    assert report.status == 'ready_for_confirmation'
    assert report.group_sizes == {'Average':2,'High':2}
    assert report.design_rank == report.design_columns == 3
    shuffled = metadata.iloc[::-1]
    again = inspect_design(shuffled, group_col='condition', numerator='High', reference='Average', covariates=('batch',))
    assert again.numerator == 'High' and again.reference == 'Average'


def test_confounded_batch_rejected():
    metadata = pd.DataFrame({'condition':['High','High','Average','Average'], 'batch':['A','A','B','B']})
    report = inspect_design(metadata, group_col='condition', numerator='High', reference='Average', covariates=('batch',))
    assert report.status == 'blocked'
    assert any('rank-deficient' in issue for issue in report.issues)


def test_multilevel_requires_explicit_subset():
    metadata = pd.DataFrame({'condition':['High','Average','Other','High','Average']})
    assert inspect_design(metadata, group_col='condition', numerator='High', reference='Average').status == 'blocked'


def test_missing_covariate_not_silently_dropped():
    metadata = pd.DataFrame({'condition':['High','High','Average','Average'], 'batch':['A',None,'B','B']})
    report = inspect_design(metadata, group_col='condition', numerator='High', reference='Average', covariates=('batch',))
    assert report.status == 'blocked'


def test_review_preserves_source_and_is_serializable():
    counts = pd.DataFrame({'S1':[1,0], 'S2':[2,0], 'S3':[3,0], 'S4':[4,0]}, index=['G1','G2'])
    metadata = pd.DataFrame({'Sample':['S1','S2','S3','S4'], 'condition':['High','High','Average','Average']})
    counts_copy = counts.copy(deep=True)
    intake = inspect_bulk_counts(counts, metadata, 'Sample')
    design = inspect_design(metadata, group_col='condition', numerator='High', reference='Average')
    review = make_intake_review(counts, metadata, intake, design)
    assert review['requires_researcher_confirmation'] is True
    assert review['source_counts_shape'] == [2,4]
    assert 'Transpose' in review['proposed_actions'][0]
    json.dumps(review)
    pd.testing.assert_frame_equal(counts, counts_copy)


def test_real_csv_and_design():
    counts = read_strict_csv(ROOT/'bc_counts_transposed_condensed.csv', use_identifier_as_index=True)
    meta = read_strict_csv(ROOT/'bc_meta.csv')
    assert counts.frame.shape == (187,110)
    assert meta.frame.shape == (187,2)
    intake = inspect_bulk_counts(counts.frame, meta.frame, 'Sample')
    assert intake.status == 'ready_for_confirmation'
    assert intake.all_zero_genes == 3
    design = inspect_design(meta.frame, group_col='condition', numerator='risk category: High', reference='risk category: Average')
    assert design.status == 'ready_for_confirmation'
    assert design.group_sizes == {'risk category: Average':112,'risk category: High':75}
    review = make_intake_review(counts.frame, meta.frame, intake, design,
                                counts_sha256=counts.sha256, metadata_sha256=meta.sha256)
    assert review['counts_sha256'] == counts.sha256
