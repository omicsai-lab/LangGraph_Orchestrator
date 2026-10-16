import copy
from pathlib import Path
import pytest
from omicsgpt.vnext_workflow import inspect_uploads, approve_review, fingerprint_config, prepare_approved_counts
ROOT = Path(__file__).resolve().parents[1]

def bundle(**settings):
    return inspect_uploads((ROOT/'bc_counts_transposed_condensed.csv').read_bytes(),
                           (ROOT/'bc_meta.csv').read_bytes(), sample_id_col='Sample',
                           group_col='condition', numerator='risk category: High',
                           reference='risk category: Average', **settings)

def test_v2_defaults_and_approval():
    b = bundle()
    assert b.config['schema_version'] == 2
    assert b.config['analysis']['min_count'] == 10
    assert b.config['analysis']['min_fraction'] == .20
    assert prepare_approved_counts(b, approve_review(b), b.config).shape == (187,110)

@pytest.mark.parametrize('key,value', [
    ('min_count',11), ('min_fraction',.3), ('dispersion_fit','mean'),
    ('refit_cooks',False), ('engine','edgeR'),
])
def test_changed_settings_invalidate_approval(key,value):
    b = bundle()
    altered = copy.deepcopy(b.config)
    altered['analysis'][key] = value
    with pytest.raises(PermissionError):
        prepare_approved_counts(b, approve_review(b), altered)

@pytest.mark.parametrize('key,value', [
    ('min_count',-1), ('min_count',1.5), ('min_count',True),
    ('min_fraction',0), ('min_fraction',float('nan')),
    ('min_fraction',1.1), ('refit_cooks','yes'), ('dispersion_fit','local'),
])
def test_invalid_settings_rejected(key,value):
    b = bundle()
    altered = copy.deepcopy(b.config)
    altered['analysis'][key] = value
    with pytest.raises(ValueError):
        fingerprint_config(altered)

def test_display_only_changes_do_not_affect_fingerprint():
    b = bundle()
    assert fingerprint_config(copy.deepcopy(b.config)) == approve_review(b)

def test_threshold_changes_change_fingerprint():
    assert approve_review(bundle(min_count=10)) != approve_review(bundle(min_count=20))
