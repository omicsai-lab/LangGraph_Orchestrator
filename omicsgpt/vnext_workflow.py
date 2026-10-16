"""Pure, fail-closed vNext bulk RNA intake and approval workflow.

No Streamlit, statistical computation, or network dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hashlib
import json
import math
import pandas as pd

from .csv_import import read_strict_csv
from .intake import inspect_bulk_counts, prepare_confirmed_counts
from .design import inspect_design


@dataclass(frozen=True)
class ReviewBundle:
    counts: Any
    metadata: Any
    intake: Any
    design: Any
    config: dict[str, Any] | None


def inspect_uploads(counts_bytes: bytes, metadata_bytes: bytes, *, sample_id_col: str,
                    group_col: str, numerator: str, reference: str,
                    covariates: tuple[str, ...] = (),
                    min_count: int = 10, min_fraction: float = 0.20,
                    dispersion_fit: str = "parametric", refit_cooks: bool = True) -> ReviewBundle:
    counts = read_strict_csv(counts_bytes, use_identifier_as_index=True)
    metadata = read_strict_csv(metadata_bytes)
    intake = inspect_bulk_counts(counts.frame, metadata.frame, sample_id_col)
    if sample_id_col not in metadata.frame.columns:
        return ReviewBundle(counts, metadata, intake, None, None)
    aligned_metadata = metadata.frame.set_index(sample_id_col, drop=False)
    design = inspect_design(aligned_metadata, group_col=group_col, numerator=numerator,
                            reference=reference, covariates=covariates)
    if intake.status != 'ready_for_confirmation' or design.status != 'ready_for_confirmation':
        return ReviewBundle(counts, metadata, intake, design, None)
    config = {
        'schema_version': 2,
        'counts_sha256': counts.sha256,
        'metadata_sha256': metadata.sha256,
        'orientation': intake.orientation,
        'sample_id_column': sample_id_col,
        'sample_order': list(metadata.frame[sample_id_col]),
        'grouping_column': group_col,
        'numerator': numerator,
        'reference': reference,
        'covariates': list(covariates),
        'transformations': [
            'transpose' if intake.orientation == 'genes_by_samples' else 'keep_orientation',
            'align_to_metadata_order',
        ],
        'analysis': {
            'engine': 'PyDESeq2',
            'filter_method': 'none_at_intake',
            'design_type': 'categorical_additive',
            'min_count': min_count,
            'min_fraction': min_fraction,
            'dispersion_fit': dispersion_fit,
            'refit_cooks': refit_cooks,
        },
    }
    return ReviewBundle(counts, metadata, intake, design, config)


def fingerprint_config(config: dict[str, Any]) -> str:
    expected = {'schema_version','counts_sha256','metadata_sha256','orientation',
                'sample_id_column','sample_order','grouping_column','numerator',
                'reference','covariates','transformations','analysis'}
    if set(config) != expected or type(config['schema_version']) is not int or config['schema_version'] != 2:
        raise ValueError('Invalid or incomplete scientific configuration (schema v2 required).')
    analysis = config['analysis']
    required_analysis = {'engine', 'filter_method', 'design_type', 'min_count',
                         'min_fraction', 'dispersion_fit', 'refit_cooks'}
    if not isinstance(analysis, dict) or set(analysis) != required_analysis:
        raise ValueError('Analysis settings must be explicit and complete.')
    if (analysis['engine'] != 'PyDESeq2' or
        analysis['filter_method'] != 'none_at_intake' or
        analysis['design_type'] != 'categorical_additive'):
        raise ValueError('Unsupported analysis settings.')
    if type(analysis['min_count']) is not int or analysis['min_count'] < 0:
        raise ValueError('min_count must be a nonnegative integer.')
    fraction = analysis['min_fraction']
    if (type(fraction) not in (int, float) or not math.isfinite(fraction)
            or not 0 < fraction <= 1):
        raise ValueError('min_fraction must be finite and in (0, 1].')
    if analysis['dispersion_fit'] != 'parametric':
        raise ValueError('Only requested parametric dispersion fit is supported in V2.4A.')
    if type(analysis['refit_cooks']) is not bool or not analysis['refit_cooks']:
        raise ValueError('Cook refitting must be enabled in the current engine.')
    for field in ('counts_sha256','metadata_sha256'):
        value = config[field]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid source hash.')
    if config['orientation'] not in ('genes_by_samples','samples_by_genes'):
        raise ValueError('Invalid orientation.')
    if not config['sample_order'] or len(set(config['sample_order'])) != len(config['sample_order']):
        raise ValueError('Invalid sample order.')
    if config['numerator'] == config['reference']:
        raise ValueError('Invalid contrast.')
    payload = json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def approve_review(bundle: ReviewBundle) -> str:
    if bundle.config is None:
        raise PermissionError('Intake and design must be valid before confirmation.')
    return fingerprint_config(bundle.config)


def prepare_approved_counts(bundle: ReviewBundle, approved_fingerprint: str,
                            current_config: dict[str, Any]) -> pd.DataFrame:
    """Revalidate approval and source inputs; return a fresh, non-mutating matrix."""
    if bundle.config is None or not approved_fingerprint:
        raise PermissionError('Researcher approval is required.')
    try:
        expected = fingerprint_config(bundle.config)
        current = fingerprint_config(current_config)
    except (ValueError, TypeError, KeyError) as exc:
        raise PermissionError('Invalid scientific configuration.') from exc
    if expected != approved_fingerprint or current != approved_fingerprint:
        raise PermissionError('Configuration changed since researcher approval.')
    if inspect_bulk_counts(bundle.counts.frame, bundle.metadata.frame,
                           bundle.config['sample_id_column']).status != 'ready_for_confirmation':
        raise PermissionError('Intake is no longer valid.')
    return prepare_confirmed_counts(bundle.counts.frame, bundle.metadata.frame,
                                    bundle.config['sample_id_column'],
                                    bundle.config['orientation'])
