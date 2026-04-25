import pandas as pd
import numpy as np
from scipy.optimize import nnls
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def run_sentinel_deconv(counts_file):
    # 1. LOAD USER DATA
    df = pd.read_csv(counts_file, index_col=0)
    if "GSM" in str(df.index[0]): df = df.T
    df.index = [str(g).split('.')[0].upper().strip() for g in df.index]
    
    # 2. THE SENTINEL MATRIX
    data = {
        'Gene':        ['EPCAM', 'KRT19', 'COL1A1', 'ACTA2', 'CD3E', 'CD8A', 'CD4', 'MS4A1', 'CD14', 'PECAM1'],
        'Tumor':       [100, 80, 0, 0, 0, 0, 0, 0, 0, 0],
        'Fibroblast':  [0, 0, 100, 70, 0, 0, 0, 0, 0, 5],
        'T_Cells':     [0, 0, 0, 0, 100, 80, 60, 0, 0, 0],
        'B_Cells':     [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
        'Myeloid':     [0, 0, 5, 0, 0, 0, 0, 0, 100, 0],
        'Endothelial': [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    }
    sig_matrix = pd.DataFrame(data).set_index('Gene')

    # 3. ALIGN & UNMIX
    common = df.index.intersection(sig_matrix.index)
    X = sig_matrix.loc[common].values
    final_results = []

    for sample in df.columns:
        y = df[sample].loc[common].values
        weights, _ = nnls(X, y)
        predicted = X.dot(weights)
        r2 = 1 - (np.sum((y - predicted)**2) / np.sum((y - np.mean(y))**2))
        
        perc = (weights / weights.sum() * 100) if weights.sum() > 0 else weights
        res = dict(zip(sig_matrix.columns, perc))
        res['Sample'] = sample
        res['R2'] = r2
        final_results.append(res)

    # RETURN the dataframe so other functions can use it
    return pd.DataFrame(final_results).set_index('Sample')

def generate_tme_heatmap(deconv_df, metadata_file):
    # 1. Load Metadata
    metadata_df = pd.read_csv(metadata_file)
    print(f"📋 Metadata Columns Found: {list(metadata_df.columns)}")

    # 2. AUTO-DISCOVER COLUMNS
    # We look for common names for Sample ID and Risk
    sample_col = next((c for c in metadata_df.columns if c.lower() in ['sample_id', 'sample', 'id', 'gsm', 'gsm_id']), None)
    risk_col = next((c for c in metadata_df.columns if c.lower() in ['risk_group', 'risk', 'condition', 'group']), None)

    if not sample_col or not risk_col:
        print("❌ Error: Could not find Sample or Risk columns.")
        print(f"Your columns are: {metadata_df.columns}")
        return

    # 3. ALIGN METADATA WITH DECONV RESULTS
    # We only want metadata for samples we actually deconvolved
    metadata_df = metadata_df[metadata_df[sample_col].isin(deconv_df.index)]
    
    # 4. COLOR CODES
    # Mapping your risk labels to colors
    risk_colors = metadata_df.set_index(sample_col)[risk_col].map({
        'High': '#e74c3c', 'Low': '#3498db', 
        'High Risk': '#e74c3c', 'Low Risk': '#3498db'
    })
    
    # 5. PREP DATA FOR PLOTTING
    plot_data = deconv_df.drop(columns=['R2'])
    
    # Ensure the order of plot_data matches risk_colors
    plot_data = plot_data.loc[risk_colors.index]

    # 6. GENERATE CLUSTERMAP
    sns.set_theme(font_scale=0.8)
    try:
        g = sns.clustermap(
            plot_data.T, 
            cmap="YlGnBu", 
            col_colors=risk_colors, 
            standard_scale=0, 
            figsize=(12, 8),
            cbar_kws={'label': 'Relative Abundance'},
            dendrogram_ratio=0.15
        )

        legend_elements = [Patch(facecolor='#e74c3c', label='High Risk'),
                           Patch(facecolor='#3498db', label='Low Risk')]
        plt.legend(handles=legend_elements, title="Clinical Risk", 
                   bbox_to_anchor=(1, 1), loc='upper left')

        plt.suptitle("TME Landscape: High vs Low Risk Cohort", y=1.02, fontsize=14)
        plt.show()
    except Exception as e:
        print(f"❌ Plotting Error: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Run deconv and capture the result in a variable
    results = run_sentinel_deconv("bc_counts_transposed.csv")
    
    # 2. Pass that variable and the filename to the heatmap
    generate_tme_heatmap(results, "bc_meta.csv")