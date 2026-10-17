"""Engineering-only GSEApy smoke test with a deliberately synthetic TEST_ONLY GMT."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from omicsgpt.gsea_contract import prepare_gsea_input, validate_ranking
from omicsgpt.gsea_execution import run_preranked_gsea, GSEASettings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        parser.error('Output directory is not empty; choose a new path')
    results = pd.read_csv(args.results, index_col='gene_id')
    ranking = validate_ranking(results)
    if len(ranking) < 30:
        parser.error('Engineering fixture requires at least 30 ranked genes')
    top = ranking.index[:12].tolist()
    bottom = ranking.index[-12:].tolist()
    gmt = ('TEST_ONLY_TOP\tSYNTHETIC_ENGINEERING_FIXTURE\t' + '\t'.join(top) + '\n'
           'TEST_ONLY_BOTTOM\tSYNTHETIC_ENGINEERING_FIXTURE\t' + '\t'.join(bottom) + '\n').encode('utf-8')
    contract = prepare_gsea_input(results, gmt, min_size=10, max_size=500,
        resource_name='TEST_ONLY_NOT_REACTOME', resource_release='engineering-fixture-v1')
    output = run_preranked_gsea(contract, settings=GSEASettings(permutation_num=1000, seed=42, threads=1))
    args.outdir.mkdir(parents=True, exist_ok=True)
    output['results'].to_csv(args.outdir / 'gsea_test_only.csv', index=False)
    (args.outdir / 'gsea_manifest.json').write_text(json.dumps(output['manifest'], indent=2), encoding='utf-8')
    print(output['results'].to_string(index=False))
    print('ENGINEERING TEST ONLY: no biological pathway claims')
    print('Output:', args.outdir)

if __name__ == '__main__':
    main()
