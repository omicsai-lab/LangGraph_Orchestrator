"""Conservative claim-aware PubMed retrieval and explicitly supported study descriptors.

No automated claim support, risk-of-bias or evidence-strength judgments.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import re
from typing import Callable
from omicsgpt.pubmed_adapter import PubMedPolicy, search_pubmed
from omicsgpt.pubmed_characterization import prepare_publication_contexts

# Only explicit vocabulary is used for bounded query expansion; no hidden LLM rewriting.
FOCUS_VOCAB = {
    'expression': 'expression', 'survival': 'survival', 'prognosis': 'prognosis',
    'predict': 'predict', 'predicts': 'predict', 'prediction': 'prediction',
    'response': 'response', 'treatment': 'treatment', 'therapy': 'therapy',
    'resistance': 'resistance', 'mutation': 'mutation', 'mutations': 'mutation',
    'metastasis': 'metastasis', 'toxicity': 'toxicity', 'knockout': 'knockout',
}
SYSTEM_PATTERNS = (
    ('animal', r'\b(mice|mouse|murine|rats|rat model)\b'),
    ('organoid', r'\borganoids?\b'),
    ('cell_line', r'\b(cell lines?|in vitro)\b'),
    ('human', r'\b(patients?|participants?|human subjects)\b'),
)
DESIGN_PATTERNS = (
    ('randomized_trial', r'\brandomi[sz]ed (?:controlled )?trial\b'),
    ('case_control', r'\bcase.control (?:study|design)\b'),
    ('cohort', r'\bcohort study\b'),
    ('cross_sectional', r'\bcross.sectional study\b'),
    ('systematic_review', r'\bsystematic review\b'),
)


def claim_focus_terms(claim: str, *, max_terms: int = 4) -> tuple[str, ...]:
    """Return only explicitly recognized concepts from the claim; never invent synonyms."""
    if not isinstance(claim, str) or not claim.strip() or claim != claim.strip():
        raise ValueError('claim must be a trimmed nonblank string')
    if type(max_terms) is not int or not 1 <= max_terms <= 5:
        raise ValueError('max_terms must be 1..5')
    found = []
    for token in re.findall(r'[A-Za-z]+', claim.lower()):
        value = FOCUS_VOCAB.get(token)
        if value and value not in found:
            found.append(value)
    return tuple(found[:max_terms])


@dataclass(frozen=True)
class ExtractedStudy:
    pmid: str
    study_system: str
    study_design: str
    system_excerpt: str | None
    design_excerpt: str | None
    claim_directness: str = 'unclear'
    finding_relation: str = 'not_assessed'
    verification: str = 'unreviewed'

    def to_dict(self) -> dict:
        return vars(self).copy()


def _explicit_label(abstract: str, patterns: tuple[tuple[str, str], ...]):
    hits = []
    for label, pattern in patterns:
        match = re.search(pattern, abstract, flags=re.IGNORECASE)
        if match:
            hits.append((label, match.group(0)))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # Different models or designs cannot be resolved by keyword matching.
        return 'mixed' if patterns is SYSTEM_PATTERNS else 'not_reported', None
    return 'not_reported', None


def extract_explicit_descriptors(context) -> ExtractedStudy:
    """Extract only literal, narrowly defined descriptors; abstain on interpretation."""
    text = context.abstract or ''
    system, system_excerpt = _explicit_label(text, SYSTEM_PATTERNS)
    design, design_excerpt = _explicit_label(text, DESIGN_PATTERNS)
    return ExtractedStudy(pmid=context.characterization.pmid, study_system=system,
        study_design=design, system_excerpt=system_excerpt, design_excerpt=design_excerpt)


def run_claim_literature(gene: str, disease: str, claim: str, *,
                         policy: PubMedPolicy = PubMedPolicy(),
                         search_fn: Callable = search_pubmed,
                         focus_terms: tuple[str, ...] | None = None) -> tuple[list[dict], dict]:
    """Search using explicit claim concepts; return separate records and an audit.

    A query without recognized concepts is allowed, but reported as not claim-focused.
    """
    terms = claim_focus_terms(claim) if focus_terms is None else focus_terms
    records, retrieval = search_fn(gene, disease, policy=policy, focus_terms=terms)
    audit = {'claim': claim, 'focus_terms': list(terms), 'claim_focused': bool(terms),
             'retrieval': retrieval, 'n_extracted': 0,
             'caution': 'Heuristic descriptors only; directness, finding direction and study quality are not assessed.'}
    if retrieval.get('status') != 'found':
        return [], audit
    contexts, context_audit = prepare_publication_contexts(records, claim=claim, disease=disease)
    result = []
    for context in contexts:
        item = context.to_dict()
        extracted = extract_explicit_descriptors(context)
        item['automated_descriptors'] = extracted.to_dict()
        result.append(item)
    audit['characterization'] = context_audit
    audit['n_extracted'] = len(result)
    return result, audit
