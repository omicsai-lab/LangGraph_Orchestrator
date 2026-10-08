"""Pure confirmation-state helpers; no Streamlit dependencies or analysis execution."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


REQUIRED_FIELDS = (
    "counts_sha256", "metadata_sha256", "orientation", "sample_id_column",
    "sample_order", "grouping_column", "numerator", "reference",
    "covariates", "transformations",
)


def _validate(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIELDS if key not in config]
    if missing:
        raise ValueError(f"Missing confirmation fields: {', '.join(missing)}")
    if any(key not in config or not isinstance(config[key], str) or len(config[key]) != 64
           or any(c not in '0123456789abcdef' for c in config[key])
           for key in ('counts_sha256', 'metadata_sha256')):
        raise ValueError('Source hashes must be lowercase SHA-256 hex strings.')
    if config['orientation'] not in ('samples_by_genes', 'genes_by_samples'):
        raise ValueError('Orientation must be resolved before confirmation.')
    if not config['grouping_column'] or not config['numerator'] or not config['reference']:
        raise ValueError('Grouping and contrast must be explicitly selected.')
    if config['numerator'] == config['reference']:
        raise ValueError('Numerator and reference must differ.')
    for key in ('sample_order', 'covariates', 'transformations'):
        if not isinstance(config[key], (tuple, list)):
            raise ValueError(f'{key} must be an ordered sequence.')
    if not config['sample_order'] or len(set(config['sample_order'])) != len(config['sample_order']):
        raise ValueError('Sample order must be nonempty and contain unique IDs.')
    if len(set(config['covariates'])) != len(config['covariates']):
        raise ValueError('Covariates must be unique.')
    # Whitelist: no unexpected fields silently influence the approved computation.
    extra = set(config) - set(REQUIRED_FIELDS)
    if extra:
        raise ValueError(f'Unexpected confirmation fields: {sorted(extra)}')
    return dict(config)


def fingerprint(config: Mapping[str, Any]) -> str:
    """Canonical fingerprint of exactly the reviewed scientific configuration."""
    validated = _validate(config)
    payload = json.dumps(validated, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def confirmation_is_current(approved_fingerprint: str | None,
                            current_config: Mapping[str, Any]) -> bool:
    """Fail closed when approval is absent or configuration is invalid/changed."""
    if not approved_fingerprint:
        return False
    try:
        return approved_fingerprint == fingerprint(current_config)
    except (TypeError, ValueError, KeyError):
        return False
