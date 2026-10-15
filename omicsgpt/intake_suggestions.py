"""Deterministic, advisory intake defaults. No scientific choices are finalized here."""
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class IntakeSuggestions:
    sample_id_column: str
    grouping_column: str | None
    sample_id_match_count: int
    sample_id_ambiguous: bool
    sample_id_reason: str
    grouping_reason: str


def suggest_columns(counts: pd.DataFrame, metadata: pd.DataFrame) -> IntakeSuggestions:
    """Rank exact sample-ID matches, then suggest a plausible categorical group column.

    This function does not modify data. User selections remain authoritative.
    """
    if metadata.empty or not len(metadata.columns):
        raise ValueError('Metadata must have at least one column and one row.')
    axes = (set(map(str, counts.index)), set(map(str, counts.columns)))
    n = len(metadata)
    candidates = []
    for col in metadata.columns:
        vals = metadata[col].astype(str).tolist()
        unique = len(set(vals)) == n and all(v.strip() for v in vals)
        matches = max(sum(v in axes[0] for v in vals), sum(v in axes[1] for v in vals))
        if unique and matches == n:
            candidates.append(col)
    if candidates:
        selected_id = candidates[0]
        id_reason = 'All metadata identifiers match one count-matrix axis.'
    else:
        # Fallback is advisory only; do not treat it as validated.
        preferred = ('sample', 'sample_id', 'sampleid', 'gsm', 'id')
        selected_id = next((c for c in metadata.columns if str(c).lower() in preferred), metadata.columns[0])
        id_reason = 'No complete unique identifier match found; please select and verify.'
    group_candidates = []
    for col in metadata.columns:
        if col == selected_id:
            continue
        vals = metadata[col].astype(str)
        k = vals.nunique(dropna=False)
        if 2 <= k < n and k <= min(20, max(2, n // 2)):
            priority = 0 if str(col).lower() in ('condition', 'group', 'treatment', 'status', 'phenotype') else 1
            group_candidates.append((priority, k, list(metadata.columns).index(col), col))
    group_candidates.sort()
    selected_group = group_candidates[0][3] if group_candidates else None
    return IntakeSuggestions(selected_id, selected_group, n if candidates else 0,
                             len(candidates) > 1, id_reason,
                             'Suggested from repeated categorical levels; verify scientific intent.' if selected_group else
                             'No clear categorical grouping variable; choose manually.')


def compact_group_sizes(sizes: dict[str, int], limit: int = 8) -> tuple[list[dict[str, object]], int]:
    """Limit UI output, while preserving the full group information in the validator."""
    ordered = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{'Group': k, 'Samples': v} for k, v in ordered[:limit]], max(0, len(ordered) - limit)
