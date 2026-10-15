"""Run the approved breast-cancer example through PyDESeq2 (no Streamlit/API)."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from omicsgpt.vnext_workflow import inspect_uploads, approve_review, prepare_approved_counts
from omicsgpt.differential_expression import FilterPolicy, run_pydeseq2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--counts', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cpus', type=int, default=1)
    args = parser.parse_args()
    counts_bytes = args.counts.read_bytes()
    metadata_bytes = args.metadata.read_bytes()
    bundle = inspect_uploads(counts_bytes, metadata_bytes,
                             sample_id_col='Sample', group_col='condition',
                             numerator='risk category: High',
                             reference='risk category: Average')
    if bundle.config is None:
        raise RuntimeError(f'Intake/design blocked: {bundle.intake.issues}; {bundle.design.issues if bundle.design else None}')
    # This script is an explicitly configured benchmark, NOT interactive researcher approval.
    # Never use this automatic benchmark approval in the production application.
    benchmark_fingerprint = approve_review(bundle)
    counts = prepare_approved_counts(bundle, benchmark_fingerprint, bundle.config)
    metadata = bundle.metadata.frame.set_index('Sample', drop=False).loc[counts.index].copy()
    policy = FilterPolicy(min_count=10, min_fraction=0.20)
    result = run_pydeseq2(counts, metadata, group_col='condition',
                          numerator='risk category: High',
                          reference='risk category: Average', policy=policy, n_cpus=args.cpus)
    table = result['results']
    required = {'log2FoldChange', 'pvalue', 'padj'}
    if not required.issubset(table.columns):
        raise RuntimeError(f'Missing required results columns: {sorted(required - set(table.columns))}')
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / 'differential_expression.csv', index_label='gene_id')
    manifest = {
        'run_type': 'noninteractive_benchmark_not_researcher_approval',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(),
        'pandas_version': pd.__version__,
        'counts_sha256': hashlib.sha256(counts_bytes).hexdigest(),
        'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'config_fingerprint': benchmark_fingerprint,
        'samples': len(counts), 'genes_input': counts.shape[1],
        'filter_audit': result['filter_audit'],
        'model': result['model'],
        'results_rows': len(table),
        'nonmissing_log2fc': int(table['log2FoldChange'].notna().sum()),
        'nonmissing_pvalue': int(table['pvalue'].notna().sum()),
        'nonmissing_padj': int(table['padj'].notna().sum()),
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    print(f'Results saved to {args.outdir}')


if __name__ == '__main__':
    main()
