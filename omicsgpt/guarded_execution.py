"""Fail-closed handoff from an approved vNext review to PyDESeq2."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from .design import inspect_design
from .differential_expression import FilterPolicy, run_pydeseq2
from .vnext_workflow import fingerprint_config, prepare_approved_counts


class ExecutionNotAuthorized(PermissionError):
    """The current scientific state is not authorized for model execution."""


def execute_approved_analysis(bundle, approved_fingerprint: str,
                              current_configuration: Mapping, *, n_cpus: int = 1,
                              engine=None):
    """Verify approval and scientific state before executing the selected model.

    ``engine`` is an injectable test seam; production callers must omit it.
    CPU count is execution provenance, not a scientific configuration field.
    """
    if bundle.config is None or not approved_fingerprint:
        raise ExecutionNotAuthorized('Valid researcher approval is required.')
    try:
        config = deepcopy(dict(current_configuration))
        current_fp = fingerprint_config(config)
        bundle_fp = fingerprint_config(bundle.config)
        if current_fp != approved_fingerprint or bundle_fp != approved_fingerprint:
            raise ExecutionNotAuthorized('Approval does not match current scientific settings.')
        counts = prepare_approved_counts(bundle, approved_fingerprint, config)
        sample_id_col = config['sample_id_column']
        source_metadata = bundle.metadata.frame
        if source_metadata[sample_id_col].tolist() != config['sample_order']:
            raise ExecutionNotAuthorized('Metadata sample order changed after approval.')
        metadata = source_metadata.set_index(sample_id_col, drop=False).loc[counts.index].copy(deep=True)
        if not metadata.index.equals(counts.index):
            raise ExecutionNotAuthorized('Metadata alignment does not match approved counts.')
        design = inspect_design(metadata, group_col=config['grouping_column'],
                                numerator=config['numerator'], reference=config['reference'],
                                covariates=tuple(config['covariates']))
        if design.status != 'ready_for_confirmation':
            raise ExecutionNotAuthorized(f'Experimental design is invalid: {design.issues}')
        settings = config['analysis']
        if (settings['engine'] != 'PyDESeq2' or
                settings['dispersion_fit'] != 'parametric' or
                settings['refit_cooks'] is not True or
                settings['design_type'] != 'categorical_additive'):
            raise ExecutionNotAuthorized('Requested model settings are unsupported.')
        policy = FilterPolicy(min_count=settings['min_count'],
                              min_fraction=settings['min_fraction'])
    except ExecutionNotAuthorized:
        raise
    except (ValueError, TypeError, KeyError, PermissionError, IndexError) as exc:
        raise ExecutionNotAuthorized('Approval or input validation failed; analysis was not started.') from exc

    # No alternate engine is ever attempted after this call.
    selected_engine = run_pydeseq2 if engine is None else engine
    return selected_engine(counts, metadata, group_col=config['grouping_column'],
                           numerator=config['numerator'], reference=config['reference'],
                           covariates=tuple(config['covariates']), policy=policy,
                           n_cpus=n_cpus)
