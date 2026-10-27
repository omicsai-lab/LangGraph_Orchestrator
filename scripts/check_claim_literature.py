"""Bounded live smoke test for claim-aware retrieval; no AI or quality inference."""
import argparse
import json
from pathlib import Path
from omicsgpt.claim_literature import run_claim_literature
from omicsgpt.pubmed_adapter import PubMedPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', default='BRCA1')
    parser.add_argument('--disease', default='breast cancer')
    parser.add_argument('--claim', default='BRCA1 expression predicts treatment response')
    parser.add_argument('--max-records', type=int, default=3)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    output, audit = run_claim_literature(args.gene, args.disease, args.claim,
        policy=PubMedPolicy(mode='fast', max_records=args.max_records))
    if audit['retrieval']['status'] != 'found':
        raise RuntimeError(f"Retrieval did not succeed: {audit['retrieval']['status']}")
    target = Path(args.outdir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / 'claim_literature.json'
    if path.exists():
        raise FileExistsError(f'Output already exists: {path}')
    path.write_text(json.dumps({'audit': audit, 'publications': output}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Query:', audit['retrieval']['query'])
    print('Records:', len(output))
    for item in output:
        print(item['characterization']['pmid'], item['automated_descriptors']['study_system'],
              item['automated_descriptors']['study_design'], 'finding=not_assessed')
    print('Saved:', path)

if __name__ == '__main__':
    main()
