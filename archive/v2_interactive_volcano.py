import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

st.set_page_config(page_title="OmicsAI - V2 Data Engine", layout="wide")

st.title("🌋 OmicsAI V2: Interactive Data Engine")
st.info("Test your real data here! Adjust the sliders to see how the statistical thresholds change the AI's targets.")

col1, col2 = st.columns([1, 2]) # 1/3 width for controls, 2/3 for the plot

with col1:
    st.subheader("1. Data Upload")
    counts_file = st.file_uploader("Upload RNA Counts", type=["csv"])
    metadata_file = st.file_uploader("Upload Metadata", type=["csv"])

    st.markdown("---")
    st.subheader("2. Statistical Cutoffs")
    # THE NEW SLIDERS
    pval_thresh = st.number_input("P-Value Cutoff", min_value=0.0001, max_value=0.1000, value=0.0500, step=0.0100, format="%.4f")
    log2fc_thresh = st.slider("Log2FC Threshold (Absolute)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    top_n_genes = st.slider("Max Targets for AI Report", min_value=1, max_value=15, value=5)

with col2:
    st.subheader("3. Interactive Volcano Plot")
    
    if counts_file and metadata_file:
        counts_df = pd.read_csv(counts_file, index_col=0)
        metadata_df = pd.read_csv(metadata_file, index_col=0)
        
        with st.spinner("Calculating Differential Expression..."):
            dds = DeseqDataSet(counts=counts_df, metadata=metadata_df, design_factors="condition")
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=["condition", "Tumor", "Normal"])
            stat_res.summary()
            results_df = stat_res.results_df
            
        # Data Prep for Plotting
        plot_df = results_df.dropna(subset=['padj', 'log2FoldChange']).copy()
        plot_df['-log10(padj)'] = -np.log10(plot_df['padj'] + 1e-300)
        plot_df['Log2FC'] = plot_df['log2FoldChange'].round(2)
        plot_df['P-value (adj)'] = plot_df['padj'].apply(lambda x: f"{x:.2e}")
        
        # Step 1: Baseline Significance using the SLIDERS
        conditions = [
            (plot_df['padj'] < pval_thresh) & (plot_df['log2FoldChange'] > log2fc_thresh), 
            (plot_df['padj'] < pval_thresh) & (plot_df['log2FoldChange'] < -log2fc_thresh)
        ]
        choices = ['Upregulated', 'Downregulated']
        plot_df['Significance'] = np.select(conditions, choices, default='Not Significant')
        
        # Step 2: Highlight the actual targets the AI will use (The "Gold Stars")
        # We sort by p-value, grab only the upregulated ones, and take the top N
        ai_targets = plot_df[plot_df['Significance'] == 'Upregulated'].sort_values(by='padj').head(top_n_genes).index
        plot_df.loc[ai_targets, 'Significance'] = 'AI Selected Target'

        # Build the Plot
        fig = px.scatter(
            plot_df, x='log2FoldChange', y='-log10(padj)', color='Significance', 
            color_discrete_map={
                'AI Selected Target': '#FFD700', # Gold!
                'Upregulated': '#EF553B',        # Red
                'Downregulated': '#636EFA',      # Blue
                'Not Significant': '#E5ECF6'     # Light Grey
            },
            hover_name=plot_df.index, 
            hover_data={'log2FoldChange': False, '-log10(padj)': False, 'Log2FC': True, 'P-value (adj)': True}
        )
        
        # ADD THE REQUESTED THRESHOLD LINES ("BARS")
        fig.add_hline(y=-np.log10(pval_thresh), line_dash="dash", line_color="black", annotation_text=f"p={pval_thresh}")
        fig.add_vline(x=log2fc_thresh, line_dash="dash", line_color="black", annotation_text=f"FC={log2fc_thresh}")
        fig.add_vline(x=-log2fc_thresh, line_dash="dash", line_color="black", annotation_text=f"FC=-{log2fc_thresh}")
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the payload that *would* go to LangGraph
        st.markdown("---")
        if len(ai_targets) > 0:
            formatted_genes = ", ".join([f"`{gene}`" for gene in ai_targets])
            st.success(f"✅ **Ready to send {len(ai_targets)} targets to the AI Orchestrator:** {formatted_genes}")
        else:
            st.warning("⚠️ **No targets selected.** Your current statistical cutoffs are too strict. Try lowering the Log2FC or increasing the P-Value.")
        
    else:
        st.info("👈 Upload data to generate the interactive plot.")