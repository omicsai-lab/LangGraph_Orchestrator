import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt
from gseapy.plot import gseaplot

print("1. Generating mock RNA-seq ranked data...")
# Spike some real genes at the top so KEGG guarantees a pathway hit
real_genes = ["TP53", "BRCA1", "EGFR", "MYC", "PTEN", "PIK3CA", "AKT1", "MTOR", "KRAS", "BRAF"]
# Fill the rest with 1000 random genes
genes = real_genes + [f"RANDOM_GENE_{i}" for i in range(1000)]

# Give the real genes artificially high fold-changes, and randomize the rest
fold_changes = [5.0, 4.8, 4.5, 4.2, 4.0, 3.8, 3.5, 3.2, 3.0, 2.8] + list(np.random.randn(1000))

# Create the ranked series format that gseapy expects
df = pd.DataFrame({"gene": genes, "rank_metric": fold_changes})
rnk = df.set_index("gene")

print("2. Running fast local GSEA...")
try:
    pre_res = gp.prerank(
        rnk=rnk,
        gene_sets='KEGG_2021_Human',
        threads=1,
        min_size=1, # Lowered so it catches our tiny spike
        max_size=1000,
        permutation_num=10, # Kept super low for a 1-second test
        seed=42
    )
    
    # Grab the very first pathway it calculated (ignoring P-value, just testing the plotter)
    terms = list(pre_res.results.keys())
    if not terms:
        print("❌ No pathways evaluated. Try again.")
    else:
        term = terms[0]
        print(f"3. Attempting to plot Mountain Plot for: {term}")
        
        # --- THE EXACT FIX WE ARE TESTING ---
        res_dict = pre_res.results[term]
        
        # Handle case-sensitivity in the dictionary for the Running Enrichment Score
        res_array = res_dict.get('RES') if 'RES' in res_dict else res_dict.get('res')
        
        # Pass the exact arguments manually
        axes = gseaplot(
            rank_metric=pre_res.ranking, 
            term=term,
            hits=res_dict['hits'],
            nes=res_dict['nes'],
            pval=res_dict['pval'],
            fdr=res_dict['fdr'],
            RES=res_array
        )
        
        # --- THE FIX: Extract the parent Figure from the list of Axes ---
        if isinstance(axes, list):
            fig = axes[0].figure  # Grab the figure from the first subplot
        elif hasattr(axes, 'figure'):
            fig = axes.figure     # If it's a single axis
        else:
            fig = plt.gcf()       # Fallback: grab the current active figure
            
        # Save it to your folder
        save_path = "test_mountain_plot.png"
        fig.savefig(save_path, bbox_inches='tight')
        print(f"✅ SUCCESS! Plot successfully saved as '{save_path}' in your current directory.")
        
except Exception as e:
    print(f"🚨 FAILED TO PLOT: {str(e)}")