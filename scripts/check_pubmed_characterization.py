"""Optional bounded live integration check; does not characterize scientific findings."""
from __future__ import annotations
import argparse
from pathlib import Path
from omicsgpt.pubmed_adapter import search_pubmed, PubMedPolicy
from omicsgpt.pubmed_characterization import prepare_publication_contexts, export_contexts_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gene', default='BRCA1')
    p.add_argument('--disease', default='breast cancer')
    p.add_argument('--claim', default='BRCA1 expression predicts treatment response')
    p.add_argument('--outdir', required=True)
    p.add_argument('--max-records', type=int, default=3)
    args = p.parse_args()
    records, retrieval = search_pubmed(args.gene, args.disease,
        policy=PubMedPolicy(mode='fast', max_records=args.max_records))
    print('Retrieval status:', retrieval['status'])
    if retrieval['status'] != 'found':
        raise RuntimeError(f"PubMed retrieval not successful: {retrieval['status']}")
    contexts, audit = prepare_publication_contexts(records, claim=args.claim, disease=args.disease)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / 'publication_contexts.json'
    if path.exists():
        raise FileExistsError(f'Output exists: {path}')
    path.write_text(export_contexts_json(contexts, {'retrieval': retrieval, 'characterization': audit}), encoding='utf-8')
    print('Publications:', audit['n_publications'], 'Abstracts:', audit['abstracts_available'])
    print('Saved:', path)
    print('No study design, finding direction, or quality inferred.')


if __name__ == '__main__':
    main()
