import pandas as pd
import numpy as np

# --- 1. Generate Transcriptomics (RNA-seq) Data ---
# Metadata (Which samples are Tumor vs Normal)
metadata = {
    'Sample': ['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4', 'Sample_5', 'Sample_6'],
    'condition': ['Normal', 'Normal', 'Normal', 'Tumor', 'Tumor', 'Tumor']
}
df_meta = pd.DataFrame(metadata).set_index('Sample')

# Raw RNA-seq Counts
np.random.seed(42) # For reproducibility
counts = {
    'TP53': np.random.randint(100, 200, 6),
    'BRCA1': np.random.randint(50, 150, 6),
    'PTEN': np.random.randint(200, 300, 6),
    'BRAF': [10, 15, 12, 800, 950, 890],  # Overexpressed in Tumor
    'EGFR': [20, 25, 22, 600, 750, 710],  # Overexpressed in Tumor
    'MYC': np.random.randint(300, 400, 6)
}
df_counts = pd.DataFrame(counts, index=['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4', 'Sample_5', 'Sample_6'])

df_meta.to_csv('mock_metadata.csv')
df_counts.to_csv('mock_counts.csv')

# --- 2. Generate Genomics (DNA Mutation) Data ---
dna_data = {
    'Gene': ['BRAF', 'KRAS', 'PIK3CA'],
    'Alteration': ['V600E', 'G12C', 'H1047R']
}
df_dna = pd.DataFrame(dna_data)
df_dna.to_csv('mock_dna.csv', index=False)

print("✅ Success! Generated: mock_metadata.csv, mock_counts.csv, and mock_dna.csv")