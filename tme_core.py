import pandas as pd
import numpy as np
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu

class TMECore:
    def __init__(self, counts_df, meta_df):
        self.counts = self._standardize_counts(counts_df)
        self.meta = meta_df
        self.markers = pd.read_csv("final_hierarchy_markers.csv")
        self.results = None
        self.stats = None

    def _standardize_counts(self, df):
        # Handle transposition and indexing
        if "GSM" in str(df.index[0]): df = df.T
        df.index = [str(g).split('.')[0].upper().strip() for g in df.index]
        return df

    def run_analysis(self):
        # 1. Align and Build Reference Matrix A
        sig_genes = list(set([g for col in self.markers.columns for g in self.markers[col].dropna()]))
        counts_sub = self.counts.loc[sig_genes]
        A = pd.DataFrame({
            group: self.counts.loc[sig_genes].apply(lambda row: 100 if row.name in self.markers[group].dropna().tolist() else 0, axis=1)
            for group in self.markers.columns
        })

        # 2. Deconvolution (NNLS)
        results = []
        for sample in counts_sub.columns:
            weights, _ = nnls(A.values, counts_sub[sample].values)
            fractions = (weights / weights.sum() * 100) if weights.sum() > 0 else weights
            res = dict(zip(A.columns, fractions))
            res['Sample_ID'] = sample
            results.append(res)
        
        self.results = pd.DataFrame(results)
        self._run_stats()
        return self.results, self.stats

    def _run_stats(self):
        # Dynamic group detection
        sample_col = next((c for c in self.meta.columns if c.lower() in ['sample_id', 'gsm', 'sample']), None)
        risk_col = next((c for c in self.meta.columns if c.lower() in ['risk_group', 'condition', 'risk']), None)
        
        groups = self.meta[risk_col].unique()
        if len(groups) < 2: return
        
        final_df = self.results.merge(self.meta[[sample_col, risk_col]], left_on='Sample_ID', right_on=sample_col)
        
        stats_list = []
        for cell_type in ['Malignant', 'Stroma', 'Immune']:
            g1_vals = final_df[final_df[risk_col] == groups[0]][cell_type]
            g2_vals = final_df[final_df[risk_col] == groups[1]][cell_type]
            _, p_val = mannwhitneyu(g1_vals, g2_vals)
            stats_list.append({
                'Cell_Type': cell_type,
                f'{groups[0]}_Median': g1_vals.median(),
                f'{groups[1]}_Median': g2_vals.median(),
                'P_Value': p_val
            })
        self.stats = pd.DataFrame(stats_list)