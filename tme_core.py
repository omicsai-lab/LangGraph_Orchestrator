import pandas as pd
import numpy as np
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu

class TMECore:
    def __init__(self, counts_df, meta_df):
        self.counts = self._standardize_data(counts_df)
        self.meta = self._standardize_meta(meta_df)
        self.signature_matrix = None # Upgraded: Will hold continuous biological data
        self.results = None

    def _standardize_data(self, df):
        # Transpose if Genes are columns
        test_genes = ['EPCAM', 'CD8A', 'COL1A1', 'CD19', 'PECAM1']
        col_hits = sum(1 for g in test_genes if g in [str(c).upper() for c in df.columns])
        if col_hits > 0: df = df.T
        
        df.index = [str(g).split('|')[-1].split('.')[0].upper().strip() for g in df.index]
        return df.groupby(level=0).sum()

    def _standardize_meta(self, df):
        df.index = [str(i).upper().replace('GSM', '').strip() for i in df.index]
        return df

    def fetch_reference_matrix(self):
        """
        Future Gatherer Module: This will ping the CZ CELLxGENE API to pull 
        actual mean expression values for Breast Cancer single-cell populations.
        """
        print("🌐 Fetching Continuous Reference Matrix...")
        
        # MOCKUP: Simulating a real continuous reference matrix (A)
        # In reality, these numbers are mean gene expression values from scRNA-seq
        mock_data = {
            'Malignant': {'KRT19': 8.5, 'EPCAM': 7.2, 'MUC1': 9.1, 'CD8A': 0.1, 'COL1A1': 0.5},
            'Stroma':    {'KRT19': 0.2, 'EPCAM': 0.1, 'MUC1': 0.1, 'CD8A': 0.1, 'COL1A1': 9.8},
            'Immune':    {'KRT19': 0.1, 'EPCAM': 0.1, 'MUC1': 0.2, 'CD8A': 8.9, 'COL1A1': 0.1}
        }
        
        self.signature_matrix = pd.DataFrame(mock_data).fillna(0.1)
        
        # Intersect signature matrix genes with bulk RNA-seq genes
        common_genes = self.signature_matrix.index.intersection(self.counts.index)
        self.signature_matrix = self.signature_matrix.loc[common_genes]
        self.counts = self.counts.loc[common_genes]
        print(f"✅ Intersected {len(common_genes)} reference genes with bulk data.")

    def run_analysis(self):
        self.fetch_reference_matrix()
        
        # --- THE UPGRADE: PER-SAMPLE NORMALIZATION ---
        # 1. Convert to relative abundance (proxy for Transcripts Per Million)
        counts_cpm = self.counts.div(self.counts.sum(axis=0), axis=1) * 1e6
        
        # 2. Log1p transform (log(x + 1)) to handle massive expression outliers
        counts_norm = np.log1p(counts_cpm)
        
        # Also log-transform the signature matrix to match mathematical spaces
        A = np.log1p(self.signature_matrix)

        results = []
        for sample in counts_norm.columns:
            b = counts_norm[sample].values
            
            # Run Non-Negative Least Squares (Ax = b)
            weights, residual = nnls(A.values, b)
            
            # UPGRADE: Do not force exactly to 100%. Normalize by a biological scalar
            # or allow relative fractions. For now, we calculate relative proportions 
            # based on the sum of weights, but leave room for an "Uncharacterized" fraction.
            total_weight = weights.sum()
            fractions = (weights / total_weight) if total_weight > 0 else weights
            
            res = dict(zip(A.columns, fractions))
            res['Sample_ID'] = str(sample).upper().replace('GSM', '').strip()
            
            # Calculate Model Confidence (R-Squared)
            predicted = A.values @ weights
            ss_res = np.sum((b - predicted)**2)
            ss_tot = np.sum((b - np.mean(b))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            res['R2'] = max(0, min(1, r2)) 
            results.append(res)

        self.results = pd.DataFrame(results)
        return self._run_stats()

    def _run_stats(self):
        risk_col = next((c for c in self.meta.columns if any(k in c.lower() for k in ['risk', 'condition', 'category', 'group', 'cohort'])), None)
        
        if not risk_col:
            return self.results, pd.DataFrame(columns=['Cell_Type', 'P_Value'])

        final_df = self.results.merge(self.meta[[risk_col]], left_on='Sample_ID', right_index=True, how='inner')
        final_df = final_df.dropna(subset=[risk_col])
        groups = final_df[risk_col].unique()
        
        if len(groups) < 2: 
            return final_df, pd.DataFrame(columns=['Cell_Type', 'P_Value'])

        stats_list = []
        for cell in self.signature_matrix.columns:
            if cell not in final_df.columns:
                continue
                
            g1 = final_df[final_df[risk_col] == groups[0]][cell].values
            g2 = final_df[final_df[risk_col] == groups[1]][cell].values
            
            if len(g1) == 0 or len(g2) == 0:
                continue
                
            _, p = mannwhitneyu(g1, g2, alternative='two-sided')
            stats_list.append({'Cell_Type': cell, 'P_Value': p})
            
        stats_df = pd.DataFrame(stats_list) if stats_list else pd.DataFrame(columns=['Cell_Type', 'P_Value'])
        return final_df, stats_df