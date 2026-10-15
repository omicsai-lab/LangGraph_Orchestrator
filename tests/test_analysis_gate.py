import copy
import hashlib
import pytest
from omicsgpt.analysis_gate import Approval, AnalysisNotAuthorized, approve, require_authorization


def config():
    return dict(counts_sha256=hashlib.sha256(b'counts').hexdigest(),
                metadata_sha256=hashlib.sha256(b'meta').hexdigest(),
                orientation='samples_by_genes', sample_id_column='Sample',
                sample_order=['S1', 'S2'], grouping_column='condition',
                numerator='High', reference='Average', covariates=[],
                transformations=['keep orientation', 'align to metadata order'])


def test_explicit_confirmation_required():
    with pytest.raises(AnalysisNotAuthorized):
        approve(config(), researcher_confirmed=False)


def test_matching_configuration_authorized():
    current = config()
    require_authorization(approve(current, researcher_confirmed=True), current)


@pytest.mark.parametrize('field,value', [
    ('counts_sha256', hashlib.sha256(b'new').hexdigest()),
    ('metadata_sha256', hashlib.sha256(b'new metadata').hexdigest()),
    ('orientation', 'genes_by_samples'),
    ('sample_order', ['S2', 'S1']),
    ('grouping_column', 'group'),
    ('numerator', 'Average'),
    ('reference', 'High'),
    ('covariates', ['batch']),
    ('transformations', ['transpose counts']),
    ('sample_id_column', 'other_id'),
])
def test_changed_scientific_input_revokes_approval(field, value):
    before = config()
    approved = approve(before, researcher_confirmed=True)
    after = copy.deepcopy(before)
    after[field] = value
    with pytest.raises(AnalysisNotAuthorized):
        require_authorization(approved, after)


def test_missing_approval_fails_closed():
    with pytest.raises(AnalysisNotAuthorized):
        require_authorization(None, config())


def test_invalid_configuration_fails_closed():
    original = config()
    approved = approve(original, researcher_confirmed=True)
    bad = dict(original)
    bad.pop('sample_order')
    with pytest.raises(AnalysisNotAuthorized):
        require_authorization(approved, bad)


def test_display_only_setting_does_not_invalidate():
    current = config()
    approval = approve(current, researcher_confirmed=True)
    display_preferences = {'theme': 'dark', 'expanded_preview': True}
    assert display_preferences['expanded_preview']
    require_authorization(approval, current)
