"""Compare independent R DESeq2 and PyDESeq2 outputs; never alter either analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")


def load_table(path: Path):
    frame = pd.read_csv(path, dtype={"gene_id": str})
    if "gene_id" not in frame or frame["gene_id"].isna().any() or frame["gene_id"].duplicated().any():
        raise ValueError(f"Invalid or duplicate gene IDs: {path}")
    missing = set(COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame.set_index("gene_id").loc[:, COLS].sort_index()


def compare(py_dir: Path, r_dir: Path, atol: float = 1e-6, rtol: float = 1e-4):
    py = load_table(py_dir / "differential_expression.csv")
    r = load_table(r_dir / "deseq2_results.csv")
    py_manifest = json.loads((py_dir / "run_manifest.json").read_text(encoding="utf-8"))
    r_manifest = json.loads((r_dir / "model_diagnostics.json").read_text(encoding="utf-8"))
    py_genes, r_genes = set(py.index), set(r.index)
    common = sorted(py_genes & r_genes)
    audit_py = py_manifest["filter_audit"]
    audit_r = r_manifest["filter_audit"]
    excluded_match = set(audit_py["excluded_genes"]) == set(audit_r["excluded_genes"])
    matched_filter = (py_genes == r_genes and excluded_match and
                      audit_py["min_samples"] == audit_r["min_samples"] and
                      audit_py["genes_before"] == audit_r["genes_before"])
    if not common:
        raise ValueError("No common genes between R and Python outputs")
    details = {}
    for col in COLS:
        a = py.loc[common, col].to_numpy(dtype=float)
        b = r.loc[common, col].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        mismatch_missing = int(np.sum(np.isfinite(a) != np.isfinite(b)))
        abs_diff = np.abs(a[both] - b[both])
        tol = np.isclose(a[both], b[both], atol=atol, rtol=rtol)
        details[col] = {
            "paired_finite": int(both.sum()),
            "missing_pattern_mismatches": mismatch_missing,
            "max_absolute_difference": float(abs_diff.max()) if abs_diff.size else None,
            "median_absolute_difference": float(np.median(abs_diff)) if abs_diff.size else None,
            "within_tolerance_count": int(tol.sum()),
            "outside_tolerance_count": int((~tol).sum()),
            "pearson_correlation": (float(np.corrcoef(a[both], b[both])[0, 1])
                                    if both.sum() >= 3 and np.std(a[both]) > 0 and np.std(b[both]) > 0 else None),
        }
    lfc_a = py.loc[common, "log2FoldChange"].to_numpy(dtype=float)
    lfc_b = r.loc[common, "log2FoldChange"].to_numpy(dtype=float)
    valid_lfc = np.isfinite(lfc_a) & np.isfinite(lfc_b)
    # Zero is a separate direction category, not positive or negative.
    direction_disagree = int(np.sum(np.sign(lfc_a[valid_lfc]) != np.sign(lfc_b[valid_lfc])))
    adj_a = py.loc[common, "padj"].to_numpy(dtype=float)
    adj_b = r.loc[common, "padj"].to_numpy(dtype=float)
    comparable = np.isfinite(adj_a) & np.isfinite(adj_b)
    sig_disagree = int(np.sum((adj_a[comparable] < 0.05) != (adj_b[comparable] < 0.05)))
    return {
        "comparison_type": "independent_R_DESeq2_vs_PyDESeq2",
        "interpretation": "Diagnostic comparison only; numerical differences require scientific review.",
        "python_genes": len(py_genes), "r_genes": len(r_genes), "common_genes": len(common),
        "python_only_genes": sorted(py_genes - r_genes), "r_only_genes": sorted(r_genes - py_genes),
        "filter_gene_sets_identical": matched_filter,
        "python_excluded_genes": len(audit_py["excluded_genes"]),
        "r_excluded_genes": len(audit_r["excluded_genes"]),
        "design": {"python": py_manifest["model"]["design"], "r": r_manifest["model"]["design"]},
        "contrast": {"python": [py_manifest["model"]["numerator"], py_manifest["model"]["reference"]],
                     "r": [r_manifest["model"]["numerator"], r_manifest["model"]["reference"]]},
        "r_dispersion_fit_used": r_manifest["model"].get("fit_type_used"),
        "r_replaced_genes": r_manifest["model"].get("replaced_genes", []),
        "alpha": 0.05,
        "lfc_direction_disagreements": direction_disagree,
        "comparable_fdr_values": int(comparable.sum()),
        "fdr_classification_disagreements": sig_disagree,
        "columns": details,
        "tolerances": {"absolute": atol, "relative": rtol},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-dir", type=Path, required=True)
    parser.add_argument("--r-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.python_dir, args.r_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["filter_gene_sets_identical"]:
        raise SystemExit("FAIL: R/Python filtering gene sets differ; investigate before interpreting estimates")


if __name__ == "__main__":
    main()
