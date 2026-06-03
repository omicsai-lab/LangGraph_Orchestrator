import requests
import pandas as pd
import time

def build_sc_signature_matrix(gene_symbols, target_tissue_filter=None):
    """
    Dynamically builds a Single-Cell Signature Matrix by querying the HPA API.
    Rows = Genes, Columns = Cell Types.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    matrix_data = {}
    
    print(f"📡 Initializing Single-Cell API fetch for {len(gene_symbols)} genes...")
    
    for gene in gene_symbols:
        print(f"  -> Fetching {gene}...")
        try:
            # 1. Get the Ensembl ID mapping
            id_url = f"https://www.proteinatlas.org/api/search_download.php?search={gene}&format=json&columns=g,eg&compress=no"
            id_res = requests.get(id_url, headers=headers)
            
            if id_res.status_code != 200:
                print(f"     [!] Failed to connect for {gene}")
                continue
                
            id_data = id_res.json()
            ensembl_id = None
            
            for entry in id_data:
                if entry.get("Gene", "").upper() == gene.upper():
                    ensembl_val = entry.get("Ensembl")
                    ensembl_id = ensembl_val[0] if isinstance(ensembl_val, list) else ensembl_val
                    break
            
            if not ensembl_id:
                print(f"     [!] No Ensembl ID found for {gene}")
                continue
                
            # 2. Fetch the detailed single-cell RNA profile
            sc_url = f"https://www.proteinatlas.org/{ensembl_id}.json"
            sc_res = requests.get(sc_url, headers=headers)
            
            if sc_res.status_code == 200:
                master_data = sc_res.json()
                entry = master_data[0] if isinstance(master_data, list) else master_data
                
                # Dig into the JSON to find the single-cell expression dictionary
                for key, val in entry.items():
                    if "single cell type specific" in key.lower() and isinstance(val, dict):
                        # Save the cell types and their expression (nTPM) for this gene
                        matrix_data[gene] = {cell.strip(): float(expr) for cell, expr in val.items()}
                        break
                        
            time.sleep(0.5) # Polite API delay to prevent IP bans
            
        except Exception as e:
            print(f"     [!] Error fetching {gene}: {e}")

    print("\n🔨 Assembling the Matrix...")
    
    # 3. Convert the nested dictionary into a Pandas DataFrame
    # This automatically aligns Genes as Rows and Cell Types as Columns!
    df = pd.DataFrame.from_dict(matrix_data, orient='index')
    
    # Fill missing values with 0 (meaning that gene isn't expressed in that cell type)
    df = df.fillna(0.0)
    
    # Optional: Filter columns to only include cell types relevant to our target tissue
    # E.g., If breast cancer, maybe we don't need 'brain glial cells' in the matrix.
    if target_tissue_filter:
        print(f"🧹 Filtering cell types for keyword: '{target_tissue_filter}'")
        relevant_cols = [c for c in df.columns if target_tissue_filter.lower() in c.lower() or "macrophage" in c.lower() or "t-cell" in c.lower()]
        if relevant_cols:
            df = df[relevant_cols]
            
    return df

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # A mix of immune, stromal, and epithelial marker genes to test the matrix spread
    test_genes = ["CD68", "CD3E", "EPCAM", "VIM", "COL1A1", "ERBB2", "MKI67", "CSF1R"]
    
    print("🚀 Starting Sandbox Test...\n")
    start_time = time.time()
    
    # Run the function (Let's filter for Breast tissue + core immune cells)
    sc_matrix = build_sc_signature_matrix(test_genes, target_tissue_filter="breast")
    
    end_time = time.time()
    
    print("\n✅ API Fetch Complete!")
    print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
    print(f"📊 Matrix Shape: {sc_matrix.shape[0]} Genes x {sc_matrix.shape[1]} Cell Types")
    print("\nPreview of the Dynamic Signature Matrix (nTPM values):")
    print(sc_matrix.head())
    
    # Save to a local CSV so you can inspect it in Excel
    sc_matrix.to_csv("sandbox_sc_matrix.csv")
    print("\n💾 Saved as 'sandbox_sc_matrix.csv' in your current folder.")