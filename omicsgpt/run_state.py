"""Framework-independent state transitions for approved analyses."""
from __future__ import annotations
from copy import deepcopy

APPROVAL_KEY = 'vnext_approved_fp'
RESULT_KEY = 'vnext_result_record'


def reconcile_state(state, current_fingerprint: str | None) -> None:
    """Remove approval and results that do not belong to the current configuration."""
    if not current_fingerprint or state.get(APPROVAL_KEY) != current_fingerprint:
        state.pop(APPROVAL_KEY, None)
    record = state.get(RESULT_KEY)
    if not isinstance(record, dict) or record.get('fingerprint') != current_fingerprint:
        state.pop(RESULT_KEY, None)


def record_approval(state, fingerprint: str) -> None:
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError('Approval fingerprint is required.')
    if state.get(APPROVAL_KEY) != fingerprint:
        state.pop(RESULT_KEY, None)
    state[APPROVAL_KEY] = fingerprint


def is_approved(state, fingerprint: str) -> bool:
    return bool(fingerprint) and state.get(APPROVAL_KEY) == fingerprint


def record_completed(state, fingerprint: str, result: dict) -> None:
    if not is_approved(state, fingerprint):
        raise PermissionError('Current configuration is not approved.')
    if not isinstance(result, dict) or not {'results', 'filter_audit', 'model', 'diagnostics'} <= result.keys():
        raise ValueError('Incomplete statistical result.')
    state[RESULT_KEY] = {'fingerprint': fingerprint, 'result': deepcopy(result)}


def current_result(state, fingerprint: str):
    if not is_approved(state, fingerprint):
        return None
    record = state.get(RESULT_KEY)
    if isinstance(record, dict) and record.get('fingerprint') == fingerprint:
        return record['result']
    return None


def clear_result(state) -> None:
    state.pop(RESULT_KEY, None)
