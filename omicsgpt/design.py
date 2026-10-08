"""Explicit, inspectable two-group additive design validation; no model fitting."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DesignReport:
    status: str
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    grouping_column: str
    numerator: str
    reference: str
    covariates: tuple[str, ...]
    group_sizes: dict[str, int]
    design_rank: int | None
    design_columns: int | None
    formula_display: str


def inspect_design(metadata: pd.DataFrame, *, group_col: str, numerator: str,
                   reference: str, covariates: tuple[str, ...] = ()) -> DesignReport:
    """Check an explicitly chosen pairwise contrast with categorical additive covariates.

    Other observed group levels are not silently removed: researcher must
    supply a predeclared subset or use a later explicit multi-level workflow.
    """
    issues: list[str] = []
    warnings: list[str] = []
    covariates = tuple(covariates)
    fields = (group_col,) + covariates
    if len(set(fields)) != len(fields):
        issues.append('Grouping variable and covariates must be distinct.')
    if any(field not in metadata.columns for field in fields):
        issues.append('Selected grouping variable or covariate is missing from metadata.')
    if not numerator or not reference or numerator == reference:
        issues.append('Choose distinct, nonblank numerator and reference levels.')
    sizes: dict[str, int] = {}
    rank = n_columns = None
    if not issues:
        if metadata[list(fields)].isna().any().any() or metadata[list(fields)].astype(str).apply(lambda x: x.str.strip().eq('')).any().any():
            issues.append('Missing/blank design values: resolve explicitly; rows were not dropped.')
        else:
            group = metadata[group_col].astype(str)
            observed = set(group.unique())
            sizes = {k: int((group == k).sum()) for k in sorted(observed)}
            if numerator not in observed or reference not in observed:
                issues.append('Numerator/reference level not present in grouping variable.')
            if observed != {numerator, reference}:
                issues.append('Additional group levels detected: explicitly subset before this two-group analysis.')
            if not issues:
                if min(sizes.values()) < 2:
                    issues.append('Each group requires at least two biological samples for this workflow.')
                columns = [np.ones(len(metadata)), (group == numerator).to_numpy(dtype=float)]
                for field in covariates:
                    values = metadata[field].astype(str)
                    if values.nunique() < 2:
                        warnings.append(f'Covariate {field!r} has one level and provides no adjustment.')
                    for level in sorted(values.unique())[1:]:
                        columns.append((values == level).to_numpy(dtype=float))
                matrix = np.column_stack(columns)
                rank, n_columns = int(np.linalg.matrix_rank(matrix)), int(matrix.shape[1])
                if rank < n_columns:
                    issues.append('Design matrix is rank-deficient; check confounding/redundant covariates.')
                if len(metadata) <= n_columns:
                    issues.append('No residual degrees of freedom for this design.')
    return DesignReport('blocked' if issues else 'ready_for_confirmation', tuple(issues),
                        tuple(warnings), group_col, numerator, reference, covariates,
                        sizes, rank, n_columns, ' + '.join(('~ 1', group_col) + covariates))
