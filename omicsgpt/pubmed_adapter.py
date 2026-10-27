"""Bounded PubMed E-Utilities retrieval; no LLM relevance or quality inference."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
import xml.etree.ElementTree as ET
from typing import Callable

import requests

from omicsgpt.evidence_schema import EvidenceRecord, make_status_record

BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
PMID_RE = re.compile(r'^[0-9]+$')


@dataclass(frozen=True)
class PubMedPolicy:
    mode: str = 'fast'
    max_records: int | None = None
    timeout_seconds: int = 15

    def __post_init__(self):
        if self.mode not in ('fast', 'deep_dive'):
            raise ValueError('mode must be fast or deep_dive')
        cap = 20 if self.mode == 'fast' else 100
        if self.max_records is not None and (type(self.max_records) is not int or not 1 <= self.max_records <= cap):
            raise ValueError(f'max_records must be between 1 and {cap} for this mode')
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 60:
            raise ValueError('timeout_seconds must be an integer between 1 and 60')

    @property
    def limit(self) -> int:
        return self.max_records or (20 if self.mode == 'fast' else 100)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ''
    return ' '.join(''.join(node.itertext()).split())


def _date(article: ET.Element) -> str | None:
    # Prefer electronic publication date, then PubMed history; avoid guessing missing dates.
    for path in ('.//ArticleDate', './/PubMedPubDate', './/JournalIssue/PubDate'):
        node = article.find(path)
        if node is None:
            continue
        year = node.findtext('Year')
        month = node.findtext('Month')
        day = node.findtext('Day')
        if year:
            return '-'.join([year] + ([month] if month else []) + ([day] if day else []))
    return None


def _parse_xml(raw: bytes, requested: set[str]) -> dict[str, dict]:
    root = ET.fromstring(raw)
    if root.tag != 'PubmedArticleSet':
        raise ValueError('Unexpected PubMed XML root')
    records = {}
    for article in root.findall('PubmedArticle'):
        pmid = _text(article.find('./MedlineCitation/PMID'))
        if not PMID_RE.fullmatch(pmid) or pmid not in requested or pmid in records:
            raise ValueError('Unexpected, invalid or duplicate PMID in PubMed response')
        title = _text(article.find('.//ArticleTitle'))
        if not title:
            raise ValueError(f'PubMed record {pmid} has no title')
        abstract_parts = []
        for section in article.findall('.//Abstract/AbstractText'):
            label = section.attrib.get('Label')
            value = _text(section)
            if value:
                abstract_parts.append((f'{label}: ' if label else '') + value)
        records[pmid] = {'pmid': pmid, 'title': title, 'abstract': '\n'.join(abstract_parts) or None,
                         'publication_date_raw': _date(article)}
    return records


def search_pubmed(
    gene: str, disease: str, *, policy: PubMedPolicy = PubMedPolicy(),
    session: requests.Session | None = None, clock: Callable[[], str] = _now,
) -> tuple[list[EvidenceRecord], dict]:
    """Search a literal gene+disease pair; return records and full query audit.

    API failures return a status record, never an empty success or fabricated evidence.
    """
    if not isinstance(gene, str) or not gene.strip() or gene != gene.strip():
        raise ValueError('gene must be a nonblank trimmed string')
    if not isinstance(disease, str) or not disease.strip() or disease != disease.strip():
        raise ValueError('disease must be a nonblank trimmed string')
    if not isinstance(policy, PubMedPolicy):
        raise TypeError('policy must be PubMedPolicy')
    # Quotes and escaping prevent user text from being interpreted as arbitrary Entrez syntax.
    def literal(s: str) -> str:
        return '"' + s.replace('"', ' ') + '"[Title/Abstract]'
    query = f'({literal(gene)}) AND ({literal(disease)})'
    timestamp = clock()
    client = session or requests.Session()
    audit = {'source': 'NCBI PubMed E-Utilities', 'mode': policy.mode, 'query': query,
             'max_records': policy.limit, 'retrieved_at_utc': timestamp, 'pmids_returned': [],
             'returned_count': 0, 'truncated_by_limit': None,
             'note': 'Retrieval only; no relevance or evidence-quality determination.'}

    def failure(status: str, message: str):
        audit['status'] = status
        audit['error'] = message
        return [make_status_record(source='PubMed', source_record_id='query:' + sha256(query.encode()).hexdigest()[:16],
                                   kind='literature', status=status, subject_id=gene, subject_id_type='gene_symbol',
                                   organism='Homo sapiens', retrieved_at_utc=timestamp,
                                   limitations=(message,))], audit

    try:
        response = client.get(BASE + '/esearch.fcgi', params={'db': 'pubmed', 'term': query,
                              'retmode': 'json', 'retmax': policy.limit, 'sort': 'relevance'},
                              timeout=policy.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        result = payload['esearchresult']
        ids = result['idlist']
        total = int(result['count'])
        if not isinstance(ids, list) or any(not isinstance(x, str) or not PMID_RE.fullmatch(x) for x in ids):
            raise ValueError('Invalid PubMed ID list')
        if len(ids) != len(set(ids)) or len(ids) > policy.limit:
            raise ValueError('Duplicate or excessive PubMed IDs')
        audit['total_matches_reported'] = total
        audit['truncated_by_limit'] = total > len(ids)
        audit['pmids_returned'] = ids
        if not ids:
            if total != 0:
                raise ValueError('PubMed reported matches but returned no IDs')
            audit['status'] = 'not_found'
            return failure('not_found', 'Search completed with zero matching PubMed records')
        response = client.get(BASE + '/efetch.fcgi', params={'db': 'pubmed', 'id': ','.join(ids),
                              'retmode': 'xml'}, timeout=policy.timeout_seconds)
        response.raise_for_status()
        raw = response.content
        fetched = _parse_xml(raw, set(ids))
        if set(fetched) != set(ids):
            raise ValueError('PubMed fetch did not return all requested PMIDs')
        audit['status'] = 'found'
        audit['returned_count'] = len(ids)
        records = []
        for pmid in ids:
            item = fetched[pmid]
            records.append(EvidenceRecord(source='PubMed', source_record_id=pmid,
                kind='literature', status='found', subject_id=gene, subject_id_type='gene_symbol',
                organism='Homo sapiens', retrieved_at_utc=timestamp,
                source_url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/', disease_context=disease,
                direction='context_only', basis='literature_metadata',
                finding=f'Publication retrieved: {item["title"]}',
                underlying_study_ids=(f'PMID:{pmid}',),
                source_payload_sha256=sha256(raw).hexdigest(),
                limitations=('Search match does not establish direct gene involvement or evidence quality.',
                             'Publication may discuss other species or contexts; requires review.'),
                attributes={**item, 'search_query': query, 'retrieval_mode': policy.mode}))
        return records, audit
    except (requests.exceptions.RequestException,) as exc:
        return failure('unavailable', f'PubMed request failed: {type(exc).__name__}')
    except (ValueError, KeyError, TypeError, ET.ParseError, json.JSONDecodeError) as exc:
        return failure('error', f'PubMed response validation failed: {type(exc).__name__}: {exc}')
