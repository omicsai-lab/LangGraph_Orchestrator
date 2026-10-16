"""Read-only provenance extraction for fitted PyDESeq2 0.5.4 datasets.

This module never modifies a fitted model or infers missing diagnostics as zero.
"""
from importlib.metadata import version


class ModelDiagnosticsError(ValueError):
    """A required diagnostic is unavailable or inconsistent."""


def _gene_flags(dds, column):
    """Return ordered flagged gene identifiers; None means attribute unavailable."""
    if column not in dds.var.columns:
        return None
    values = dds.var[column]
    if values.isna().any():
        raise ModelDiagnosticsError(f"Diagnostic {column!r} contains missing flags")
    if not values.map(lambda v: isinstance(v, (bool, __import__('numpy').bool_))).all():
        raise ModelDiagnosticsError(f"Diagnostic {column!r} must contain boolean flags")
    return [str(gene) for gene, flagged in zip(dds.var.index, values) if flagged]


def extract_model_diagnostics(dds):
    """Extract requested/actual dispersion fit and Cook's refit provenance.

    Requires a completed PyDESeq2 0.5.4 fit. Unsupported or absent essential
    fitted state raises instead of fabricating a result.
    """
    requested = getattr(dds, 'fit_type', None)
    uns = getattr(dds, 'uns', None)
    if requested not in ('parametric', 'mean') or uns is None:
        raise ModelDiagnosticsError('Requested dispersion fit is unavailable or unsupported')
    actual = uns.get('disp_function_type')
    if actual not in ('parametric', 'mean'):
        raise ModelDiagnosticsError('Actual dispersion fit is unavailable or unsupported')
    if not hasattr(dds, 'var') or not dds.var.index.is_unique:
        raise ModelDiagnosticsError('Fitted gene identifiers are unavailable or duplicated')
    replaced = _gene_flags(dds, 'replaced')
    refitted = _gene_flags(dds, 'refitted')
    refit_enabled = getattr(dds, 'refit_cooks', None)
    if not isinstance(refit_enabled, bool):
        raise ModelDiagnosticsError('Cook refit setting is unavailable')
    if refit_enabled and replaced is None:
        raise ModelDiagnosticsError('Cook refitting enabled but replaced flags unavailable')
    if replaced is not None and refitted is not None and not set(refitted).issubset(set(replaced)):
        raise ModelDiagnosticsError('Refitted genes are not a subset of replaced genes')
    return {
        'pydeseq2_version': version('pydeseq2'),
        'dispersion_fit_requested': requested,
        'dispersion_fit_actual': actual,
        'dispersion_fallback_occurred': requested != actual,
        'refit_cooks_enabled': refit_enabled,
        'replaced_genes': replaced,
        'replaced_genes_count': None if replaced is None else len(replaced),
        'refitted_genes': refitted,
        'refitted_genes_count': None if refitted is None else len(refitted),
    }
