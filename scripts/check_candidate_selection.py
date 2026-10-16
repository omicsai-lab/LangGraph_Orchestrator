"""Exploratory real-data validation of V2.5A candidate selection."""

from pathlib import Path

import pandas as pd

from omicsgpt.results_explorer import (
    SelectionPolicy,
    select_candidates,
    prepare_volcano,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "benchmark_outputs/v2_2/differential_expression.csv"

results = pd.read_csv(RESULTS_PATH, index_col="gene_id")

print("=" * 65)
print("OMICSGPT V2.5A — REAL-DATA CANDIDATE SELECTION")
print("=" * 65)

print(f"Genes analyzed: {len(results)}")
print(f"FDR < 0.05: {(results['padj'] < 0.05).sum()}")

strategies = [
    "balanced",
    "significance",
    "effect",
]

selected_sets = {}

for strategy in strategies:
    policy = SelectionPolicy(
        strategy=strategy,
        n=5,
        fdr=0.05,
        min_abs_log2fc=0.0,
    )

    genes, audit = select_candidates(results, policy)
    selected_sets[strategy] = set(genes)

    selected = results.loc[genes].copy()

    positive = int((selected["log2FoldChange"] > 0).sum())
    negative = int((selected["log2FoldChange"] < 0).sum())

    print(f"\nSTRATEGY: {strategy}")
    print(f"Selected genes: {len(genes)}")
    print(f"Positive log2FC: {positive}")
    print(f"Negative log2FC: {negative}")

    print(
        selected[
            ["log2FoldChange", "pvalue", "padj"]
        ].to_string()
    )

    print("Selection audit:", audit)

    # Verify deterministic behavior
    genes_repeat, audit_repeat = select_candidates(results, policy)

    assert genes == genes_repeat
    assert audit == audit_repeat

print("\n" + "=" * 65)
print("OVERLAP BETWEEN SELECTION STRATEGIES")
print("=" * 65)

for i, strategy_a in enumerate(strategies):
    for strategy_b in strategies[i + 1:]:
        overlap = selected_sets[strategy_a] & selected_sets[strategy_b]

        print(
            f"{strategy_a} vs {strategy_b}: "
            f"{len(overlap)} overlapping genes"
        )

print("\n" + "=" * 65)
print("VOLCANO DATA VALIDATION")
print("=" * 65)

volcano, volcano_audit = prepare_volcano(
    results,
    p_column="pvalue",
    fdr=0.05,
)

print("Volcano rows:", len(volcano))
print("Volcano audit:", volcano_audit)

print("\nPASS: Candidate selection completed deterministically.")