import copy
from pathlib import Path
import pytest
from omicsgpt.vnext_workflow import inspect_uploads, approve_review, prepare_approved_counts

ROOT = Path(__file__).resolve().parents[1]


def load_bundle():
    return inspect_uploads((ROOT / 'bc_counts_transposed_condensed.csv').read_bytes(),
                           (ROOT / 'bc_meta.csv').read_bytes(),
                           sample_id_col='Sample', group_col='condition',
                           numerator='risk category: High', reference='risk category: Average')


def test_real_data_approved_handoff():
    bundle = load_bundle()
    assert bundle.intake.status == 'ready_for_confirmation'
    assert bundle.design.status == 'ready_for_confirmation'
    assert bundle.design.group_sizes == {'risk category: Average':112,'risk category: High':75}
    approved = approve_review(bundle)
    prepared = prepare_approved_counts(bundle, approved, bundle.config)
    assert prepared.shape == (187,110)
    assert prepared.index.tolist() == bundle.metadata.frame['Sample'].tolist()


@pytest.mark.parametrize('change', [
    lambda c: c.update(numerator='risk category: Average'),
    lambda c: c['analysis'].update(engine='RPKM/T-Test'),
    lambda c: c['analysis'].update(filter_method='low_counts'),
    lambda c: c.update(transformations=['transpose']),
    lambda c: c.update(sample_order=list(reversed(c['sample_order']))),
    lambda c: c.update(metadata_sha256='0'*64),
])
def test_changes_invalidate_approval(change):
    bundle = load_bundle()
    approved = approve_review(bundle)
    config = copy.deepcopy(bundle.config)
    change(config)
    with pytest.raises(PermissionError):
        prepare_approved_counts(bundle, approved, config)


def test_missing_approval_rejected():
    bundle = load_bundle()
    with pytest.raises(PermissionError):
        prepare_approved_counts(bundle, '', bundle.config)


def test_same_file_different_contrast_changes_fingerprint():
    bundle = load_bundle()
    other = inspect_uploads((ROOT / 'bc_counts_transposed_condensed.csv').read_bytes(),
                           (ROOT / 'bc_meta.csv').read_bytes(), sample_id_col='Sample',
                           group_col='condition', numerator='risk category: Average',
                           reference='risk category: High')
    assert approve_review(bundle) != approve_review(other)
