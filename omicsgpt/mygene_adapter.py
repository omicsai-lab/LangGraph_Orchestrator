"""Audited human gene-symbol lookup via MyGene.info; no inference or silent mapping."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Any

import requests

from omicsgpt.evidence_schema import EvidenceRecord, make_status_record

ENDPOINT = 'https://mygene.info/v3/query'
_FIELDS = 'symbol,name,summary,type_of_gene,alias,entrezgene,ensembl,uniprot,taxid'


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status(symbol: str, status: str, timestamp: str, reason: str) -> EvidenceRecord:
    return make_status_record(source='MyGene.info', source_record_id=f'query:symbol:{symbol}',
                              kind='gene_annotation', status=status, subject_id=symbol,
                              subject_id_type='gene_symbol', organism='Homo sapiens',
                              retrieved_at_utc=timestamp, limitations=(reason,))


def lookup_human_gene(symbol: str, *, session: Any = None, timeout: float = 10.0,
                      retrieved_at_utc: str | None = None) -> EvidenceRecord:
    """Exact, case-sensitive symbol match with explicit ambiguity and HTTP failure statuses.

    `session` permits deterministic offline testing. Aliases never establish identity.
    """
    if not isinstance(symbol, str) or not symbol or symbol != symbol.strip() or not re.fullmatch(r'[A-Za-z0-9_.-]+', symbol):
        raise ValueError('Provide one nonblank gene symbol containing letters, numbers, _, . or -')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 60:
        raise ValueError('timeout must be between 0 and 60 seconds')
    timestamp = retrieved_at_utc or _utc_now()
    client = session if session is not None else requests
    try:
        response = client.get(ENDPOINT, params={'q': f'symbol:{symbol}', 'species': 'human',
                                               'fields': _FIELDS, 'size': 100}, timeout=timeout)
        if response.status_code != 200:
            status = 'unavailable' if response.status_code in (429, 500, 502, 503, 504) else 'error'
            return _status(symbol, status, timestamp, f'HTTP {response.status_code}; no annotation accepted')
        payload = response.json()
    except (requests.RequestException, TimeoutError, OSError) as exc:
        return _status(symbol, 'unavailable', timestamp, f'Network request failed: {type(exc).__name__}')
    except (ValueError, TypeError):
        return _status(symbol, 'error', timestamp, 'Invalid JSON response')
    if not isinstance(payload, dict) or not isinstance(payload.get('hits'), list):
        return _status(symbol, 'error', timestamp, 'Unexpected MyGene.info response structure')
    # A truncated result page cannot establish uniqueness.
    hits = payload['hits']
    if payload.get('total', len(hits)) > len(hits):
        return _status(symbol, 'error', timestamp, 'Search results truncated; uniqueness not established')
    matches = [hit for hit in hits if isinstance(hit, dict) and
               hit.get('symbol') == symbol and hit.get('taxid') == 9606]
    ids = {str(hit.get('_id', '')) for hit in matches}
    if len(matches) > 1 or len(ids) > 1:
        return _status(symbol, 'error', timestamp, 'Ambiguous exact human-symbol matches; manual review required')
    if not matches:
        return _status(symbol, 'not_found', timestamp, 'No exact human gene-symbol match; aliases not substituted')
    hit = matches[0]
    if not hit.get('_id'):
        return _status(symbol, 'error', timestamp, 'Matched record lacks stable source identifier')
    aliases = hit.get('alias', [])
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
        return _status(symbol, 'error', timestamp, 'Unexpected alias format')
    attrs = {key: hit[key] for key in ('symbol', 'name', 'summary', 'type_of_gene', 'entrezgene', 'ensembl', 'uniprot') if key in hit}
    attrs['aliases'] = aliases
    attrs['taxid'] = 9606
    try:
        canonical = json.dumps(hit, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        digest = sha256(canonical).hexdigest()
        return EvidenceRecord(source='MyGene.info', source_record_id=str(hit['_id']),
                              kind='gene_annotation', status='found', subject_id=symbol,
                              subject_id_type='gene_symbol', organism='Homo sapiens',
                              retrieved_at_utc=timestamp,
                              source_url=f'https://mygene.info/v3/gene/{hit["_id"]}',
                              direction='context_only', basis='aggregated_evidence',
                              finding=f'Exact human gene-symbol annotation for {symbol}',
                              limitations=('Gene annotation is contextual, not disease association or target validation.',
                                           'Source aggregates upstream databases; verify original sources for consequential claims.'),
                              source_payload_sha256=digest, attributes=attrs)
    except (TypeError, ValueError):
        return _status(symbol, 'error', timestamp, 'Annotation cannot be represented as a validated evidence record')
