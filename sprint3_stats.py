import pandas as pd
import numpy as np
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu

def run_dynamic_discovery():
    print("🧬 PHASE 3.1: DYNAMIC CLINICAL DISCOVERY")
    print("-" * 65)

    # 1. LOAD DATA & SIGNATURES
    counts = pd.read_csv("bc_counts_transposed.csv", index_col=0)
    if "GSM" in str(counts.index[0]): counts = counts.T
    counts.index = [str(g).split('.')[0].upper().strip() for g in counts.index]
    
    sentinels = pd.read_csv("final_hierarchy_markers.csv")
    metadata = pd.read_csv("bc_meta.csv")
    
    # 2. ALIGN COORDINATES
    sig_genes = list(set([g for col in sentinels.columns for g in sentinels[col].dropna()]))
    counts_sub = counts.loc[sig_genes]
    
    # Build Signature Matrix A (Genes x CellTypes)
    A_df = pd.DataFrame({
        group: counts.loc[sig_genes].apply(lambda row: 100 if row.name in sentinels[group].dropna().tolist() else 0, axis=1)
        for group in sentinels.columns
    })

    # 3. UNMIXING (NNLS)
    results = []
    for sample in counts_sub.columns:
        weights, _ = nnls(A_df.values, counts_sub[sample].values)
        fractions = (weights / weights.sum() * 100) if weights.sum() > 0 else weights
        res = dict(zip(A_df.columns, fractions))
        res['Sample_ID'] = sample
        results.append(res)
    deconv_df = pd.DataFrame(results)

    # 4. DYNAMIC GROUP DETECTION
    sample_col = next((c for c in metadata.columns if c.lower() in ['sample_id', 'gsm', 'sample']), None)
    risk_col = next((c for c in metadata.columns if c.lower() in ['risk_group', 'condition', 'risk']), None)
    
    # Identify unique groups
    groups = metadata[risk_col].unique()
    if len(groups) < 2:
        print(f"❌ Error: Found only one group ({groups}). Comparison requires two."); return
    
    g1_name, g2_name = groups[0], groups[1]
    print(f"✅ Dynamic Groups Identified: '{g1_name}' vs '{g2_name}'")

    # Merge results
    final_df = deconv_df.merge(metadata[[sample_col, risk_col]], left_on='Sample_ID', right_on=sample_col)

    # 5. STATISTICAL RIGOR
    stats_report = []
    for cell_type in ['Malignant', 'Stroma', 'Immune']:
        g1_vals = final_df[final_df[risk_col] == g1_name][cell_type]
        g2_vals = final_df[final_df[risk_col] == g2_name][cell_type]
        
        u_stat, p_val = mannwhitneyu(g1_vals, g2_vals)
        stats_report.append({
            'Cell_Type': cell_type,
            f'{g1_name}_Median': f"{g1_vals.median():.2f}%",
            f'{g2_name}_Median': f"{g2_vals.median():.2f}%",
            'P_Value': f"{p_val:.4e}",
            'Significant': '✅' if p_val < 0.05 else '❌'
        })

    # 6. FINAL REPORT
    print("\n" + "="*85)
    print(f"📊 DISCOVERY SUMMARY: {g1_name} vs {g2_name} (n={len(final_df)})")
    print("="*85)
    print(pd.DataFrame(stats_report).to_string(index=False))
    print("-" * 85)
    
    final_df.to_csv("final_deconvolution_results.csv", index=False)

if __name__ == "__main__":
    run_dynamic_discovery()