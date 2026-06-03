import pandas as pd
import requests
import io

# --- CONFIGURATION ---
USER_DATA_PATH = "bc_counts_transposed.csv"

# Updated to use more stable 2026 keywords
CELL_TYPES = {
    'Tumor':      ['EPITHELIAL', 'MALIGNANT', 'CANCER', 'MAMMARY'],
    'Fibroblast': ['FIBROBLAST', 'STROMA', 'CAF'],
    'T_Cells':    ['T CELL', 'LYMPHOCYTE', 'CD8', 'CD4'],
    'Myeloid':    ['MACROPHAGE', 'MONOCYTE', 'MYELOID', 'DENDRITIC'],
    'Endothelial':['ENDOTHELIAL', 'VASCULAR'],
    'B_Cells':    ['B CELL', 'PLASMA']
}

def fetch_gmt_robust(library):
    """Fetches gene sets with a fallback mechanism."""
    url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={library}"
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200 or len(r.text) < 100:
            return None
        sets = {}
        for line in r.text.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) > 2:
                sets[parts[0].upper()] = set([g.upper().strip() for g in parts[2:] if g])
        return sets
    except:
        return None

def run_consensus_v2():
    print("🔬 SPRINT 1.1: ROBUST CONSENSUS AUDIT")
    print("-" * 50)

    # 1. LOAD LOCAL GENE LIST
    try:
        bulk_df = pd.read_csv(USER_DATA_PATH, index_col=0, nrows=2)
        if "GSM" in str(bulk_df.index[0]): bulk_df = bulk_df.T
        local_genes = set([str(g).split('.')[0].upper().strip() for g in bulk_df.index])
        print(f"✅ Local Universe: {len(local_genes)} genes.")
    except:
        print("❌ Could not read local file."); return

    # 2. FETCH LIBRARIES (Using Azimuth 2024 as the stable partner)
    lib1 = fetch_gmt_robust("CellMarker_2024")
    lib2 = fetch_gmt_robust("Azimuth_Cell_Types_2021") # Azimuth is highly stable
    
    if not lib1 or not lib2:
        print(f"❌ API Fetch Failed. Lib1: {bool(lib1)}, Lib2: {bool(lib2)}")
        return

    print(f"📡 Libraries Online: CellMarker ({len(lib1)}), Azimuth ({len(lib2)})")

    # 3. INTERSECTION WITH RELAXED KEYWORDS
    consensus_matrix = {}
    for label, keywords in CELL_TYPES.items():
        set1 = set().union(*[v for k, v in lib1.items() if any(x in k for x in keywords)])
        set2 = set().union(*[v for k, v in lib2.items() if any(x in k for x in keywords)])
        
        # The Consensus
        overlap = set1.intersection(set2).intersection(local_genes)
        
        # Prune to top 15 (alphabetical for reproducibility)
        consensus_matrix[label] = sorted(list(overlap))[:15]
        print(f"🎯 {label:12}: {len(overlap)} consensus markers found.")

    # 4. EXPORT
    df_out = pd.DataFrame.from_dict(consensus_matrix, orient='index').T
    df_out.to_csv("consensus_sentinels.csv", index=False)
    print("-" * 50)
    
    if df_out.isna().all().all():
        print("🚨 CRITICAL: STILL 0 GENES. Checking keyword logic...")
    else:
        print("💾 Success. Foundation built in 'consensus_sentinels.csv'.")

if __name__ == "__main__":
    run_consensus_v2()