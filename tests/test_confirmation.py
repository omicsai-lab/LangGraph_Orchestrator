"""Independent state-transition tests; no Streamlit or APIs required."""
import copy
import pytest
from omicsgpt.confirmation import fingerprint, confirmation_is_current


@pytest.fixture
def config():
    return dict(counts_sha256='a'*64, metadata_sha256='b'*64,
                orientation='samples_by_genes', sample_id_column='Sample',
                sample_order=['S1', 'S2', 'S3', 'S4'], grouping_column='condition',
                numerator='High', reference='Average', covariates=['batch'],
                transformations=['align samples to metadata order'])


def test_approval_matches(config):
    assert confirmation_is_current(fingerprint(config), config)


def test_missing_approval_is_false(config):
    assert not confirmation_is_current(None, config)


@pytest.mark.parametrize('field,value', [
    ('counts_sha256', 'c'*64), ('metadata_sha256', 'd'*64),
    ('orientation', 'genes_by_samples'), ('sample_id_column', 'PatientID'),
    ('sample_order', ['S2', 'S1', 'S3', 'S4']),
    ('grouping_column', 'group'), ('numerator', 'Average'),
    ('reference', 'High'), ('covariates', []),
    ('transformations', ['transpose', 'align samples to metadata order']),
])
def test_scientific_change_revokes_approval(config, field, value):
    old = fingerprint(config)
    updated = copy.deepcopy(config)
    updated[field] = value
    assert not confirmation_is_current(old, updated)


def test_display_settings_not_in_fingerprint(config):
    original = fingerprint(config)
    assert confirmation_is_current(original, copy.deepcopy(config))


def test_invalid_orientation_blocks(config):
    config['orientation'] = 'unknown'
    with pytest.raises(ValueError):
        fingerprint(config)


def test_duplicate_samples_block(config):
    config['sample_order'] = ['S1', 'S1']
    with pytest.raises(ValueError):
        fingerprint(config)


def test_extra_field_blocks(config):
    config['unreviewed_setting'] = True
    with pytest.raises(ValueError):
        fingerprint(config)


def test_order_of_dictionary_keys_irrelevant(config):
    assert fingerprint(config) == fingerprint(dict(reversed(list(config.items()))))
