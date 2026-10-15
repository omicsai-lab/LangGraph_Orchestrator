"""Fail-closed authorization of a previously reviewed scientific configuration.

Pure functions: Streamlit and statistical engines must call this gate immediately
before execution, not merely when drawing an enabled button.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any
from .confirmation import fingerprint, confirmation_is_current


class AnalysisNotAuthorized(PermissionError):
    pass


@dataclass(frozen=True)
class Approval:
    fingerprint: str
    researcher_confirmed: bool


def approve(config: Mapping[str, Any], *, researcher_confirmed: bool) -> Approval:
    if not researcher_confirmed:
        raise AnalysisNotAuthorized('Explicit researcher confirmation is required.')
    return Approval(fingerprint(config), True)


def require_authorization(approval: Approval | None, current_config: Mapping[str, Any]) -> None:
    if approval is None or not approval.researcher_confirmed:
        raise AnalysisNotAuthorized('No approved intake and design configuration.')
    if not confirmation_is_current(approval.fingerprint, current_config):
        raise AnalysisNotAuthorized('Inputs, design, or transformations changed; re-confirm before analysis.')
