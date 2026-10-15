"""Standalone PyDESeq2 execution. Never silently changes statistical engines."""
from dataclasses import dataclass
from importlib.metadata import version
from time import perf_counter
import math
import numpy as np
import pandas as pd


class DifferentialExpressionError(RuntimeError):
    """An explicitly selected differential-expression analysis failed."""


@dataclass(frozen=True)
class FilterPolicy:
    min_count: int = 10
    min_fraction: float = 0.20

    def __post_init__(self):
        if type(self.min_count) is not int or self.min_count < 0:
            raise ValueError('min_count must be a nonnegative integer')
        if not isinstance(self.min_fraction, (int, float)) or not 0 < self.min_fraction <= 1:
            raise ValueError('min_fraction must be in (0, 1]')


def filter_counts(counts: pd.DataFrame, policy: FilterPolicy):
    """Return a new filtered matrix and immutable audit metadata; no source mutation."""
    if counts.empty or not counts.index.is_unique or not counts.columns.is_unique:
        raise ValueError('Counts must be nonempty with unique sample and gene identifiers')
    numeric = counts.apply(pd.to_numeric, errors='raise')
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not (values == np.floor(values)).all():
        raise ValueError('Counts must be finite, nonnegative integers; no correction attempted')
    n_required = max(1, math.ceil(len(counts) * policy.min_fraction))
    keep = (numeric >= policy.min_count).sum(axis=0) >= n_required
    retained = numeric.loc[:, keep].copy()
    if retained.empty:
        raise ValueError('No genes passed the specified prefilter; analysis not run')
    audit = {'min_count': policy.min_count, 'min_fraction': policy.min_fraction,
             'min_samples': n_required, 'genes_before': len(counts.columns),
             'genes_after': len(retained.columns),
             'excluded_genes': counts.columns[~keep].tolist()}
    return retained.astype(np.int64), audit


def run_pydeseq2(counts: pd.DataFrame, metadata: pd.DataFrame, *, group_col: str,
                  numerator: str, reference: str, covariates=(),
                  policy: FilterPolicy = FilterPolicy(), n_cpus: int = 1):
    """Run only PyDESeq2; explicit contrast, no fallback or imputation.

    Caller must validate experimental design and researcher authorization before invocation.
    """
    if n_cpus < 1:
        raise ValueError('n_cpus must be >= 1')
    if not counts.index.equals(metadata.index):
        raise ValueError('Count and metadata sample order must match exactly')
    if group_col not in metadata or any(c not in metadata for c in covariates):
        raise ValueError('Missing design column')
    if numerator == reference or set(metadata[group_col].astype(str)) != {numerator, reference}:
        raise ValueError('Explicit contrast must match exactly two observed groups')
    filtered, audit = filter_counts(counts, policy)
    design = list(covariates) + [group_col]
    start = perf_counter()
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        dds = DeseqDataSet(counts=filtered, metadata=metadata,
                           design='~ ' + ' + '.join(design), refit_cooks=True, n_cpus=n_cpus)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=[group_col, numerator, reference], n_cpus=n_cpus)
        stats.summary()
        results = stats.results_df.copy(deep=True)
    except Exception as exc:
        raise DifferentialExpressionError(f'PyDESeq2 failed; no alternative engine was run: {exc}') from exc
    return {'results': results, 'filter_audit': audit,
            'model': {'engine': 'PyDESeq2', 'version': version('pydeseq2'),
                      'design': design, 'numerator': numerator, 'reference': reference,
                      'n_cpus': n_cpus, 'elapsed_seconds': perf_counter() - start}}
