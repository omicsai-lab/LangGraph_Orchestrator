"""Synthetic fitted-object contract tests; actual fit must be checked in Docker."""
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest
from omicsgpt.model_diagnostics import extract_model_diagnostics, ModelDiagnosticsError


def fitted(requested='parametric', actual='mean', replaced=True):
    frame = pd.DataFrame({'replaced': [True, False, True],
                          'refitted': [True, False, True]},
                         index=['G1', 'G2', 'G3'])
    return SimpleNamespace(fit_type=requested,
                           uns={'disp_function_type': actual},
                           var=frame, refit_cooks=replaced)


@patch('omicsgpt.model_diagnostics.version', return_value='0.5.4')
def test_mean_fallback_and_replaced_genes(_):
    report = extract_model_diagnostics(fitted())
    assert report['dispersion_fallback_occurred'] is True
    assert report['replaced_genes'] == ['G1', 'G3']
    assert report['refitted_genes_count'] == 2
    assert report['pydeseq2_version'] == '0.5.4'


@patch('omicsgpt.model_diagnostics.version', return_value='0.5.4')
def test_parametric_without_fallback(_):
    assert not extract_model_diagnostics(fitted('parametric', 'parametric'))['dispersion_fallback_occurred']


@patch('omicsgpt.model_diagnostics.version', return_value='0.5.4')
def test_explicit_mean_without_fallback(_):
    assert not extract_model_diagnostics(fitted('mean', 'mean'))['dispersion_fallback_occurred']


@patch('omicsgpt.model_diagnostics.version', return_value='0.5.4')
def test_missing_refitted_distinguished_from_zero(_):
    dds = fitted()
    dds.var = dds.var.drop(columns='refitted')
    assert extract_model_diagnostics(dds)['refitted_genes_count'] is None


@pytest.mark.parametrize('change', [
    lambda d: d.uns.clear(),
    lambda d: d.var.__setitem__('replaced', [True, None, False]),
    lambda d: d.var.__setitem__('replaced', [1, 0, 1]),
    lambda d: setattr(d, 'fit_type', 'unknown'),
    lambda d: d.var.__setitem__('refitted', [False, True, False]),
])
def test_invalid_diagnostics_fail_closed(change):
    dds = fitted()
    change(dds)
    with pytest.raises(ModelDiagnosticsError):
        extract_model_diagnostics(dds)


@patch('omicsgpt.model_diagnostics.version', return_value='0.5.4')
def test_refit_disabled_can_report_no_replacement_column(_):
    dds = fitted(replaced=False)
    dds.var = dds.var.drop(columns=['replaced', 'refitted'])
    result = extract_model_diagnostics(dds)
    assert result['replaced_genes'] is None
    assert result['replaced_genes_count'] is None
