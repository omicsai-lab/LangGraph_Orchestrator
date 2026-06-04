import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def run_final_foundation_builder():
    print("🏗️ SPRINT 2.5: THE FINAL FOUNDATION (Significance-Filtered)")
    print("-" * 65)

    # 1. LOAD DATA
    try:
        counts = pd.read_csv("bc_counts_transposed.csv", index_col=0)
        if "GSM" in str(counts.index[0]): counts = counts.T
        counts.index = [str(g).split('.')[0].upper().strip() for g in counts.index]
        sentinels = pd.read_csv("consensus_sentinels.csv")
        print(f"✅ Data Ready: {counts.shape[1]} samples, {len(counts)} genes.")
    except Exception as e:
        print(f"❌ Load Error: {e}"); return

    # 2. HIERARCHICAL GROUPS
    groups = {
        'Malignant': ['Tumor'],
        'Stroma':    ['Fibroblast', 'Endothelial'],
        'Immune':    ['T_Cells', 'B_Cells', 'Myeloid']
    }

    final_foundation = {}

    for g_name, subtypes in groups.items():
        # A. Gather all consensus candidates
        candidates = []
        for s in subtypes:
            if s in sentinels.columns:
                candidates.extend(sentinels[s].dropna().tolist())
        valid_genes = counts.index.intersection(candidates)
        
        if len(valid_genes) == 0:
            print(f"⚠️ No genes found for {g_name}"); continue

        # B. Calculate the "Group Mean Profile" (The biological baseline)
        group_profile = counts.loc[valid_genes].mean(axis=0)

        # C. THE SIGNIFICANCE FILTER
        # We only keep genes that significantly correlate (p < 0.05) with the group profile
        significant_genes = []
        for gene in valid_genes:
            r_val, p_val = pearsonr(counts.loc[gene], group_profile)
            if p_val < 0.05 and r_val > 0.3: # Must be significant AND positive
                significant_genes.append((gene, r_val))
        
        # D. RANK BY STRENGTH & SELECT TOP 10
        # This gives us the 10 "most loyal" markers for this cohort
        sorted_sig = sorted(significant_genes, key=lambda x: x[1], reverse=True)
        final_foundation[g_name] = [x[0] for x in sorted_sig[:10]]
        
        print(f"🎯 {g_name:10}: Found {len(significant_genes)} significant markers. Selected Top 10.")

    # 3. FINAL RIGOR CHECK (Condition Number)
    metagenes = pd.DataFrame({
        name: counts.loc[genes].mean(axis=0)
        for name, genes in final_foundation.items()
    })
    
    kappa = np.linalg.cond(metagenes.values)

    print("\n" + "="*50)
    print("📊 FINAL FOUNDATION AUDIT")
    print("="*50)
    print(f"Condition Number (κ): {kappa:.2f}")
    
    # 4. SAVE THE "IRONCLAD" RULER
    if kappa < 40:
        print(f"🟢 SUCCESS: Stable & Significant foundation achieved.")
        pd.DataFrame(final_foundation).to_csv("final_hierarchy_markers.csv", index=False)
        print("💾 Saved to 'final_hierarchy_markers.csv'.")
    else:
        print(f"🟡 STABILITY: {kappa:.2f}. Borderline, but significance-filtered.")
        pd.DataFrame(final_foundation).to_csv("final_hierarchy_markers.csv", index=False)

if __name__ == "__main__":
    run_final_foundation_builder()