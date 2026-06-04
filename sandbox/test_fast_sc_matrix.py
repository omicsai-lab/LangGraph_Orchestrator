import pandas as pd
import time
import os

def test_hpa_local_file(tissue_filter="breast", filepath="rna_single_cell_type_tissue.tsv.zip"):
    print("🚀 Starting Local Flat File Sandbox Test...\n")
    start_time = time.time()
    
    if not os.path.exists(filepath):
        print(f"🚨 CRITICAL ERROR: Could not find '{filepath}'.")
        print("Please download it from https://www.proteinatlas.org/download/rna_single_cell_type_tissue.tsv.zip and place it in this folder.")
        return None
    
    # 1. Load into Pandas directly from the local ZIP file
    print(f"🔨 Parsing local file '{filepath}' into Pandas DataFrame...")
    df = pd.read_csv(filepath, sep='\t', compression='zip')
    parse_time = time.time()
    print(f"   [+] Parsing complete in {parse_time - start_time:.2f} seconds.")
    
    # 2. Filter by Tissue and build the matrix
    print(f"\n🧹 Filtering for Tissue: '{tissue_filter}'...")
    
    # Keep only the rows matching the requested tissue
    tissue_df = df[df['Tissue'].str.lower() == tissue_filter.lower()]
    
    if tissue_df.empty:
        print(f"⚠️ No data found for tissue '{tissue_filter}'.")
        return None
        
    # 3. Pivot the table: Genes as Rows, Cell Types as Columns
    print("🧮 Pivoting into Gene x Cell Type Matrix...")
    matrix_df = tissue_df.pivot_table(
        index='Gene name', 
        columns='Cell type', 
        values='nTPM', 
        aggfunc='mean' # Averages the expression if there are duplicates
    ).fillna(0.0)
    
    end_time = time.time()
    
    print(f"\n✅ Matrix Assembly Complete!")
    print(f"⏱️ Total Time taken: {end_time - start_time:.2f} seconds")
    print(f"📊 Final Matrix Shape: {matrix_df.shape[0]} Genes x {matrix_df.shape[1]} Cell Types")
    print(f"🧬 Cell Types Found: {list(matrix_df.columns)}")
    
    return matrix_df

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # Run the function for Breast Tissue
    matrix = test_hpa_local_file(tissue_filter="breast")
    
    if matrix is not None:
        # Test genes: Immune, Epithelial, Stromal
        test_genes = ["CD68", "CD3E", "EPCAM", "VIM", "COL1A1", "ERBB2", "MKI67", "CSF1R"]
        
        # Safely extract only the genes that actually exist in the matrix
        valid_genes = [g for g in test_genes if g in matrix.index]
        
        print("\nPreview of the Dynamic Signature Matrix (Target Genes):")
        print(matrix.loc[valid_genes])
        
        # Save the full 15,000+ gene matrix to CSV
        matrix.to_csv("fast_sandbox_sc_matrix.csv")
        print("\n💾 Saved full matrix as 'fast_sandbox_sc_matrix.csv' in your current folder.")