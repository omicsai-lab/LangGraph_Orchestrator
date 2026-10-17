"""Strict, non-fitting GSEA ranking and GMT resource validation.

No network requests, identifier conversion, or statistical inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import math
import pandas as pd


class GSEAContractError(ValueError):
    """An input cannot be used for GSEA without an explicit correction."""


@dataclass(frozen=True)
class GSEAInput:
    ranking: pd.Series
    gene_sets: dict[str, tuple[str, ...]]
    audit: dict


def validate_ranking(results: pd.DataFrame, *, column: str = "stat") -> pd.Series:
    """Validate and sort a signed gene-level ranking without changing the caller."""
    if not isinstance(results, pd.DataFrame) or results.empty:
        raise GSEAContractError("Results must be a nonempty DataFrame.")
    if column not in results.columns:
        raise GSEAContractError(f"Ranking column {column!r} is missing.")
    idx = results.index
    if not idx.is_unique or idx.hasnans or any(not isinstance(x, str) or not x.strip() for x in idx):
        raise GSEAContractError("Gene identifiers must be unique, nonblank strings.")
    if not pd.api.types.is_numeric_dtype(results[column]):
        raise GSEAContractError("Ranking statistics must have numeric dtype; no coercion attempted.")
    series = results[column]
    if series.isna().any():
        raise GSEAContractError("Missing ranking statistics; no genes were silently dropped.")
    values = series.to_numpy(dtype=float)
    if not all(math.isfinite(float(x)) for x in values):
        raise GSEAContractError("Nonfinite ranking statistics are not allowed.")
    if len(set(values)) == 1:
        raise GSEAContractError("All ranking statistics are identical; enrichment is not informative.")
    # Sorting by the numeric statistic, then the original identifier, is deterministic.
    ordered = sorted(zip(idx.tolist(), values.tolist()), key=lambda x: (-x[1], x[0]))
    return pd.Series([v for _, v in ordered], index=pd.Index([g for g, _ in ordered], name="gene_id"), name=column, dtype="float64")


def read_gmt(source: str | Path | bytes) -> tuple[dict[str, tuple[str, ...]], dict]:
    """Read a local UTF-8 GMT without guessing identifier mappings or collapsing duplicates."""
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GSEAContractError("GMT must be UTF-8.") from exc
    gene_sets: dict[str, tuple[str, ...]] = {}
    descriptions: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise GSEAContractError(f"Blank GMT line {number}.")
        fields = line.split("\t")
        if len(fields) < 3:
            raise GSEAContractError(f"GMT line {number} needs name, description, and genes.")
        name, description, *genes = fields
        if not name.strip() or name != name.strip() or any(not g.strip() or g != g.strip() for g in genes):
            raise GSEAContractError(f"Invalid GMT identifier on line {number}.")
        if name in gene_sets:
            raise GSEAContractError(f"Duplicate GMT pathway name: {name!r}.")
        if len(genes) != len(set(genes)):
            raise GSEAContractError(f"Duplicate gene membership in pathway {name!r}.")
        gene_sets[name] = tuple(genes)
        descriptions[name] = description
    if not gene_sets:
        raise GSEAContractError("GMT is empty.")
    return gene_sets, {"gmt_sha256": sha256(raw).hexdigest(), "pathways_input": len(gene_sets), "descriptions": descriptions}


def prepare_gsea_input(
    results: pd.DataFrame,
    gmt_source: str | Path | bytes,
    *,
    ranking_column: str = "stat",
    min_size: int = 10,
    max_size: int = 500,
    organism: str = "Homo sapiens",
    resource_name: str = "Reactome",
    resource_release: str | None = None,
) -> GSEAInput:
    """Build an audited input contract; does not execute enrichment."""
    if type(min_size) is not int or type(max_size) is not int or min_size < 1 or max_size < min_size:
        raise GSEAContractError("Invalid min/max gene-set sizes.")
    if not organism.strip() or not resource_name.strip() or not resource_release:
        raise GSEAContractError("Organism, resource name and release must be explicit.")
    ranking = validate_ranking(results, column=ranking_column)
    gene_sets, resource_audit = read_gmt(gmt_source)
    universe = set(ranking.index)
    covered = set()
    retained: dict[str, tuple[str, ...]] = {}
    exclusions = []
    for pathway, genes in gene_sets.items():
        overlap = tuple(g for g in genes if g in universe)
        covered.update(overlap)
        if min_size <= len(overlap) <= max_size:
            retained[pathway] = overlap
        else:
            exclusions.append({"pathway": pathway, "source_size": len(genes), "overlap_size": len(overlap), "reason": "outside_overlap_size_bounds"})
    if not retained:
        raise GSEAContractError("No gene sets meet overlap size limits; inspect mapping coverage and parameters.")
    audit = {
        "ranking_column": ranking_column,
        "ranking_genes": len(ranking),
        "organism": organism,
        "identifier_matching": "exact_case_sensitive_no_mapping",
        "resource_name": resource_name,
        "resource_release": resource_release,
        "resource_sha256": resource_audit["gmt_sha256"],
        "pathways_input": len(gene_sets),
        "pathways_retained": len(retained),
        "pathways_excluded": exclusions,
        "genes_matching_any_pathway": len(covered),
        "fraction_ranked_genes_matching_any_pathway": len(covered) / len(ranking),
        "note": "Exact matching is not gene-identifier mapping; GSEA has not been run.",
    }
    return GSEAInput(ranking=ranking, gene_sets=retained, audit=audit)
