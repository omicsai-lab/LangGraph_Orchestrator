"""Compare two independently produced OmicsGPT DE benchmark output folders.

Run after invoking scripts.benchmark_pydeseq2 twice with identical inputs.
This comparison never fits a model and never modifies source outputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ('baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj')
PROVENANCE_KEYS = ('counts_sha256', 'metadata_sha256', 'config_fingerprint', 'samples', 'genes_input')


def compare(run_a: Path, run_b: Path, *, atol: float = 1e-8, rtol: float = 1e-6, alpha: float = 0.05) -> dict:
    if run_a.resolve() == run_b.resolve():
        raise ValueError('Run directories must be different.')
    if not (0 <= alpha <= 1 and atol >= 0 and rtol >= 0):
        raise ValueError('alpha must be in [0,1]; tolerances must be nonnegative.')
    ma = json.loads((run_a / 'run_manifest.json').read_text(encoding='utf-8'))
    mb = json.loads((run_b / 'run_manifest.json').read_text(encoding='utf-8'))
    provenance_differences = [key for key in PROVENANCE_KEYS if ma.get(key) != mb.get(key)]
    if ma.get('filter_audit') != mb.get('filter_audit'):
        provenance_differences.append('filter_audit')
    for key in ('engine', 'version', 'design', 'numerator', 'reference', 'n_cpus'):
        if ma.get('model', {}).get(key) != mb.get('model', {}).get(key):
            provenance_differences.append('model.' + key)

    a = pd.read_csv(run_a / 'differential_expression.csv', index_col=0)
    b = pd.read_csv(run_b / 'differential_expression.csv', index_col=0)
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError('Duplicate gene IDs in result files.')
    if set(a.index) != set(b.index):
        raise ValueError('Gene sets differ between runs.')
    if not set(NUMERIC_COLUMNS).issubset(a.columns) or not set(NUMERIC_COLUMNS).issubset(b.columns):
        raise ValueError('Required DE result columns are missing.')
    b = b.loc[a.index]
    columns = {}
    exact_all = True
    close_all = True
    for col in NUMERIC_COLUMNS:
        x = pd.to_numeric(a[col], errors='raise').to_numpy(dtype=float)
        y = pd.to_numeric(b[col], errors='raise').to_numpy(dtype=float)
        exact = bool(np.array_equal(x, y, equal_nan=True))
        close = bool(np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True))
        finite = np.isfinite(x) & np.isfinite(y)
        max_diff = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else None
        columns[col] = {'exact': exact, 'within_tolerance': close, 'max_absolute_difference': max_diff,
                        'missing_pattern_equal': bool(np.array_equal(np.isnan(x), np.isnan(y)))}
        exact_all &= exact
        close_all &= close

    pa = pd.to_numeric(a['padj'], errors='raise')
    pb = pd.to_numeric(b['padj'], errors='raise')
    significance_changed = int(((pa < alpha) != (pb < alpha)).sum())
    fa = pd.to_numeric(a['log2FoldChange'], errors='raise')
    fb = pd.to_numeric(b['log2FoldChange'], errors='raise')
    direction_changed = int(((fa > 0) != (fb > 0)).sum())
    decision_stable = significance_changed == 0 and direction_changed == 0
    return {
        'run_a': str(run_a), 'run_b': str(run_b), 'n_genes': len(a),
        'provenance_differences': provenance_differences,
        'exact_numeric_match': exact_all, 'within_numeric_tolerance': close_all,
        'alpha': alpha, 'significance_classifications_changed': significance_changed,
        'effect_directions_changed': direction_changed,
        'decision_stable': decision_stable, 'columns': columns,
        'overall_pass': not provenance_differences and close_all and decision_stable,
        'notes': 'Agreement between repeated runs does not establish statistical validity or agreement with R DESeq2.'
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-a', required=True, type=Path)
    parser.add_argument('--run-b', required=True, type=Path)
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.run_a, args.run_b, atol=args.atol, rtol=args.rtol, alpha=args.alpha)
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + '\n', encoding='utf-8')
    raise SystemExit(0 if result['overall_pass'] else 1)


if __name__ == '__main__':
    main()
