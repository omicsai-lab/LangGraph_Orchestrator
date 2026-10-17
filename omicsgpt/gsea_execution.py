"""Audited, non-network preranked GSEApy execution over a validated GSEAInput."""
from __future__ import annotations
from dataclasses import dataclass
from importlib.metadata import version
from time import perf_counter
import hashlib
import json
import math
import pandas as pd
from .gsea_contract import GSEAInput, GSEAContractError

class GSEAExecutionError(RuntimeError):
    """GSEA execution failed; no substitute results were produced."""

@dataclass(frozen=True)
class GSEASettings:
    permutation_num: int = 1000
    seed: int = 42
    threads: int = 1
    weight: float = 1.0

    def __post_init__(self):
        if type(self.permutation_num) is not int or self.permutation_num < 1:
            raise ValueError('permutation_num must be a positive integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError('threads must be a positive integer')
        if isinstance(self.weight, bool) or not isinstance(self.weight, (float, int)) or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('weight must be finite and nonnegative')


def _validate_input(data: GSEAInput) -> None:
    if not isinstance(data, GSEAInput):
        raise GSEAContractError('Expected a validated GSEAInput object')
    rank = data.ranking
    if not isinstance(rank, pd.Series) or rank.empty or not rank.index.is_unique or rank.isna().any():
        raise GSEAContractError('Ranking must be nonempty, unique and complete')
    if not pd.api.types.is_numeric_dtype(rank) or not all(math.isfinite(float(x)) for x in rank):
        raise GSEAContractError('Ranking must contain finite numeric statistics')
    if len(set(rank.to_numpy())) == 1 or not rank.is_monotonic_decreasing:
        raise GSEAContractError('Ranking must be nonconstant and sorted descending')
    if not data.gene_sets or not isinstance(data.audit, dict) or not data.audit.get('resource_sha256'):
        raise GSEAContractError('Gene sets and resource provenance are required')
    universe = set(rank.index)
    for name, genes in data.gene_sets.items():
        if not isinstance(name, str) or not name or not genes or len(genes) != len(set(genes)) or not set(genes).issubset(universe):
            raise GSEAContractError('Invalid or unmatched pathway membership')


def run_preranked_gsea(data: GSEAInput, *, settings: GSEASettings = GSEASettings(), engine=None) -> dict:
    """Run GSEApy with no external resource retrieval; injectable engine for isolated tests.

    Caller must use prepare_gsea_input to validate original results and GMT first.
    GSEApy FDR is reported unchanged, not relabeled as BH adjusted p-values.
    """
    _validate_input(data)
    if not isinstance(settings, GSEASettings):
        raise TypeError('settings must be GSEASettings')
    if engine is None:
        try:
            import gseapy
        except ImportError as exc:
            raise GSEAExecutionError('GSEApy is not installed in this environment') from exc
        engine = gseapy.prerank
    ranking = data.ranking.copy(deep=True)
    sets = {k: list(v) for k, v in data.gene_sets.items()}
    started = perf_counter()
    try:
        fit = engine(rnk=ranking, gene_sets=sets, permutation_num=settings.permutation_num,
                     seed=settings.seed, threads=settings.threads, weight=float(settings.weight),
                     min_size=1, max_size=len(ranking), outdir=None, no_plot=True, verbose=False)
        raw = fit.res2d
    except Exception as exc:
        raise GSEAExecutionError(f'GSEApy failed: {type(exc).__name__}: {exc}') from exc
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise GSEAExecutionError('GSEApy returned no pathway results')
    required = ['Term', 'ES', 'NES', 'NOM p-val', 'FDR q-val', 'Lead_genes']
    if any(col not in raw.columns for col in required):
        raise GSEAExecutionError(f'Unexpected GSEApy result schema: {list(raw.columns)}')
    if raw['Term'].isna().any() or raw['Term'].duplicated().any() or not set(raw['Term']).issubset(sets):
        raise GSEAExecutionError('Unexpected or duplicate pathway names in GSEApy output')
    rows = raw[required].copy(deep=True).rename(columns={
        'Term': 'pathway', 'ES': 'es', 'NES': 'nes', 'NOM p-val': 'nominal_pvalue',
        'FDR q-val': 'gseapy_fdr', 'Lead_genes': 'leading_edge_genes'})
    for col in ('es', 'nes', 'nominal_pvalue', 'gseapy_fdr'):
        try:
            rows[col] = pd.to_numeric(rows[col], errors='raise')
        except (ValueError, TypeError) as exc:
            raise GSEAExecutionError(f'Invalid {col} in GSEApy results') from exc
        if not rows[col].map(math.isfinite).all():
            raise GSEAExecutionError(f'Nonfinite {col} in GSEApy results')
    for col in ('nominal_pvalue', 'gseapy_fdr'):
        if not rows[col].between(0, 1).all():
            raise GSEAExecutionError(f'Invalid probability range in {col}')
    for _, row in rows.iterrows():
        leading = row['leading_edge_genes']
        if not isinstance(leading, str):
            raise GSEAExecutionError('Leading-edge genes must be a semicolon-delimited string')
        genes = [g for g in leading.split(';') if g]
        if not set(genes).issubset(set(sets[row['pathway']])):
            raise GSEAExecutionError('Leading-edge genes do not belong to their pathway')
    rows = rows.sort_values(['gseapy_fdr', 'nominal_pvalue', 'pathway'], kind='stable').reset_index(drop=True)
    try:
        engine_version = version('gseapy')
    except Exception:
        engine_version = 'unavailable_in_test_environment'
    rank_digest = hashlib.sha256(json.dumps(list(zip(ranking.index.tolist(), ranking.tolist())), separators=(',', ':')).encode()).hexdigest()
    manifest = {'engine': 'GSEApy', 'engine_version': engine_version, 'method': 'prerank',
                'permutation_num': settings.permutation_num, 'seed': settings.seed,
                'threads': settings.threads, 'weight': float(settings.weight),
                'ranking_sha256': rank_digest, 'ranked_genes': len(ranking),
                'gene_sets_tested': len(sets), 'pathways_reported': len(rows),
                'elapsed_seconds': perf_counter() - started,
                'input_audit': dict(data.audit),
                'note': 'GSEApy FDR q-values are engine-specific; no BH substitution. No biological conclusions from synthetic/test GMT.'}
    return {'results': rows, 'manifest': manifest}
