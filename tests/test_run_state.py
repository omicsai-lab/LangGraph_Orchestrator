import pytest
from omicsgpt.run_state import (reconcile_state, record_approval, is_approved,
    record_completed, current_result, clear_result)


def result():
    return {'results': 'table', 'filter_audit': {}, 'model': {}, 'diagnostics': {}}


def test_initial_state_is_unapproved():
    assert not is_approved({}, 'a')
    assert current_result({}, 'a') is None


def test_approval_and_completion():
    s = {}
    record_approval(s, 'a')
    assert is_approved(s, 'a')
    record_completed(s, 'a', result())
    assert current_result(s, 'a')['results'] == 'table'


def test_changed_configuration_invalidates_both():
    s = {}
    record_approval(s, 'a')
    record_completed(s, 'a', result())
    reconcile_state(s, 'b')
    assert not is_approved(s, 'b')
    assert current_result(s, 'b') is None
    assert current_result(s, 'a') is None


def test_invalid_input_invalidates_both():
    s = {}
    record_approval(s, 'a')
    record_completed(s, 'a', result())
    reconcile_state(s, None)
    assert s == {}


def test_display_only_rerun_preserves_result():
    s = {}
    record_approval(s, 'a')
    record_completed(s, 'a', result())
    reconcile_state(s, 'a')
    assert current_result(s, 'a') is not None


def test_unapproved_execution_cannot_be_recorded():
    with pytest.raises(PermissionError):
        record_completed({}, 'a', result())


def test_incomplete_result_rejected():
    s = {}
    record_approval(s, 'a')
    with pytest.raises(ValueError):
        record_completed(s, 'a', {'results': 'table'})
    assert current_result(s, 'a') is None


def test_new_approval_clears_old_results():
    s = {}
    record_approval(s, 'a')
    record_completed(s, 'a', result())
    record_approval(s, 'b')
    assert current_result(s, 'b') is None


def test_clear_result_preserves_approval():
    s = {}
    record_approval(s, 'a')
    record_completed(s, 'a', result())
    clear_result(s)
    assert is_approved(s, 'a') and current_result(s, 'a') is None


def test_reject_blank_approval():
    with pytest.raises(ValueError):
        record_approval({}, '')
