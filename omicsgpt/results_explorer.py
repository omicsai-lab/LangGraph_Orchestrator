"""Deterministic, presentation-only exploration of differential-expression results."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

REQUIRED = ('log2FoldChange', 'pvalue', 'padj')
STRATEGIES = ('balanced', 'significance', 'effect', 'manual')


@dataclass(frozen=True)
class SelectionPolicy:
    strategy: str = 'balanced'
    n: int = 5
    fdr: float = 0.05
    min_abs_log2fc: float = 0.0
    manual_genes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError('Unknown selection strategy')
        if type(self.n) is not int or self.n < 1:
            raise ValueError('n must be a positive integer')
        if type(self.fdr) not in (float, int) or not math.isfinite(self.fdr) or not 0 < self.fdr <= 1:
            raise ValueError('fdr must be finite in (0, 1]')
        if type(self.min_abs_log2fc) not in (float, int) or not math.isfinite(self.min_abs_log2fc) or self.min_abs_log2fc < 0:
            raise ValueError('min_abs_log2fc must be finite and nonnegative')
        if not isinstance(self.manual_genes, tuple) or any(not isinstance(g, str) for g in self.manual_genes):
            raise ValueError('manual_genes must be a tuple of gene identifiers')


def prepare_volcano(results: pd.DataFrame, *, p_column: str = 'pvalue', fdr: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Return a fresh plotting table and an audit; no replacement of zero p-values."""
    if p_column not in ('pvalue', 'padj'):
        raise ValueError('p_column must be pvalue or padj')
    if not 0 < fdr <= 1 or not math.isfinite(fdr):
        raise ValueError('Invalid FDR cutoff')
    table = _validated(results)
    p = table[p_column]
    positive = p[(p > 0) & p.notna()]
    floor = max(float(positive.min()) / 10, np.nextafter(0.0, 1.0)) if len(positive) else None
    y = pd.Series(np.nan, index=table.index, dtype=float)
    valid = p.notna() & table['log2FoldChange'].notna()
    y.loc[valid & (p > 0)] = -np.log10(p.loc[valid & (p > 0)])
    if floor is not None:
        y.loc[valid & (p == 0)] = -np.log10(floor)
    table['volcano_y'] = y
    table['volcano_display_capped'] = valid & (p == 0)
    table['fdr_significant'] = table['padj'].notna() & (table['padj'] <= fdr)
    table['plot_eligible'] = y.notna()
    return table, {
        'p_column': p_column, 'fdr': fdr,
        'n_total': len(table), 'n_plottable': int(table['plot_eligible'].sum()),
        'n_unplottable': int((~table['plot_eligible']).sum()),
        'n_zero_p_capped': int(table['volcano_display_capped'].sum()),
        'zero_p_display_floor': floor,
        'note': 'Zero p-values are capped for plotting only; original p-values remain unchanged.'
    }


def select_candidates(results: pd.DataFrame, policy: SelectionPolicy) -> tuple[list[str], dict]:
    """Return deterministic candidate IDs; selection does not establish biomarker validity."""
    table = _validated(results)
    if policy.strategy == 'manual':
        unknown = sorted(set(policy.manual_genes) - set(table.index))
        if unknown:
            raise ValueError(f'Unknown gene identifiers: {unknown}')
        chosen = list(dict.fromkeys(policy.manual_genes))
        return chosen, {'strategy': 'manual', 'selected': len(chosen), 'unknown_genes': [],
                        'eligibility': 'Explicit researcher selection; no automatic FDR filter'}
    eligible = table.loc[table['padj'].notna() & (table['padj'] <= policy.fdr) &
                         table['log2FoldChange'].notna() &
                         (table['log2FoldChange'].abs() >= policy.min_abs_log2fc)].copy()
    eligible['gene_id'] = eligible.index
    if policy.strategy == 'balanced':
        up = eligible.loc[eligible['log2FoldChange'] > 0].sort_values(
            ['log2FoldChange', 'padj', 'gene_id'], ascending=[False, True, True], kind='stable')
        down = eligible.loc[eligible['log2FoldChange'] < 0].sort_values(
            ['log2FoldChange', 'padj', 'gene_id'], ascending=[True, True, True], kind='stable')
        chosen = up.index[:policy.n].tolist() + down.index[:policy.n].tolist()
    elif policy.strategy == 'significance':
        chosen = eligible.sort_values(['padj', 'gene_id'], kind='stable').index[:policy.n].tolist()
    else:
        eligible['abs_lfc'] = eligible['log2FoldChange'].abs()
        chosen = eligible.sort_values(['abs_lfc', 'padj', 'gene_id'],
                                      ascending=[False, True, True], kind='stable').index[:policy.n].tolist()
    return chosen, {'strategy': policy.strategy, 'fdr': policy.fdr,
                    'min_abs_log2fc': policy.min_abs_log2fc, 'n_requested': policy.n,
                    'n_eligible': len(eligible), 'n_selected': len(chosen),
                    'selected_genes': chosen,
                    'note': 'Exploratory candidate selection, not validated biomarker prioritization.'}


def _validated(results: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(results, pd.DataFrame) or results.empty or not results.index.is_unique:
        raise ValueError('Expected nonempty DE table with unique gene IDs')
    if results.index.hasnans or any(not isinstance(g, str) or not g for g in results.index):
        raise ValueError('Gene identifiers must be nonempty strings')
    if not set(REQUIRED).issubset(results.columns):
        raise ValueError(f'Missing DE columns: {sorted(set(REQUIRED) - set(results.columns))}')
    table = results.copy(deep=True)
    # Normalize only the private working copy's index name. Gene IDs and values
    # remain unchanged, and the caller's index metadata is preserved.
    table.index.name = None
    for col in REQUIRED:
        if not pd.api.types.is_numeric_dtype(table[col]):
            raise ValueError(f'{col} must be numeric')
        values = table[col].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'{col} contains nonfinite values')
    for col in ('pvalue', 'padj'):
        if ((table[col].dropna() < 0) | (table[col].dropna() > 1)).any():
            raise ValueError(f'{col} must be in [0,1]')
    return table
