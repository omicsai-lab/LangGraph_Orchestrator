"""Presentation-only Streamlit results explorer for approved DE outcomes.

No model fitting, authorization mutations, or external network requests.
"""
from __future__ import annotations

import pandas as pd
from omicsgpt.results_explorer import SelectionPolicy, prepare_volcano, select_candidates


STRATEGY_EXPLANATIONS = {
    'balanced': ('Balanced up/down',
        'Among genes meeting the exploratory FDR and effect-size criteria, select up to N with the largest positive log2FC and N with the most negative log2FC (up to 2N total). This deliberately explores both directions; it does not imply equal biological importance.'),
    'significance': ('Statistical evidence',
        'Among eligible genes, select up to N total with the smallest adjusted p-values. This emphasizes statistical evidence, not necessarily the largest expression differences.'),
    'effect': ('Effect magnitude',
        'Among eligible genes, select up to N total with the largest absolute log2FC. This emphasizes estimated expression differences, which may have uncertainty.'),
    'manual': ('Researcher-directed',
        'Highlight genes chosen by the researcher, including genes outside the automatic FDR criteria. This does not imply statistical significance or biomarker validation.'),
}


def format_probability(value: object) -> str:
    """Presentation-only p-value formatting; never changes numeric results."""
    if pd.isna(value):
        return 'NA'
    number = float(value)
    if number == 0:
        return '0 (computational)'
    if 0 < number < 0.001:
        return f'{number:.2e}'
    return f'{number:.4f}'


def display_results_table(results: pd.DataFrame) -> pd.DataFrame:
    """Create readable table strings without modifying the source numeric table."""
    shown = results.rename_axis('gene_id').reset_index().copy(deep=True)
    for col in ('pvalue', 'padj'):
        if col in shown:
            shown[col] = shown[col].map(format_probability)
    return shown


def candidate_marker_style() -> dict:
    """Visible on light/dark themes; marker shape distinguishes selection."""
    return {'symbol': 'star', 'size': 16, 'color': '#FACC15',
            'line': {'color': '#4B3B00', 'width': 1.4}}


def explorer_payload(results: pd.DataFrame, *, strategy: str = 'balanced', n: int = 5,
                     fdr: float = 0.05, min_abs_log2fc: float = 0.0,
                     manual_genes: tuple[str, ...] = (), p_column: str = 'pvalue') -> dict:
    """Build an immutable-for-caller visualization payload and selection audit."""
    policy = SelectionPolicy(strategy=strategy, n=n, fdr=fdr,
                             min_abs_log2fc=min_abs_log2fc,
                             manual_genes=manual_genes)
    selected, selection_audit = select_candidates(results, policy)
    volcano, volcano_audit = prepare_volcano(results, p_column=p_column, fdr=fdr)
    volcano = volcano.copy(deep=True)
    volcano['gene_id'] = volcano.index.to_list()
    volcano['candidate_selected'] = volcano.index.isin(selected)
    volcano['category'] = 'Not FDR significant'
    volcano.loc[volcano['fdr_significant'] & (volcano['log2FoldChange'] > 0), 'category'] = 'Higher in numerator'
    volcano.loc[volcano['fdr_significant'] & (volcano['log2FoldChange'] < 0), 'category'] = 'Lower in numerator'
    volcano.loc[volcano['fdr_significant'] & (volcano['log2FoldChange'] == 0), 'category'] = 'FDR significant, zero effect'
    return {'volcano': volcano, 'volcano_audit': volcano_audit,
            'selected_genes': selected, 'selection_audit': selection_audit}


def render_results_panel(results: pd.DataFrame, *, numerator: str, reference: str,
                         fingerprint: str) -> None:
    """Render explorer with display-only widgets; never touches approved run state."""
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader('Explore differential expression')
    st.caption(f'Comparison: {numerator} versus {reference}. Positive log2FC means higher expression in {numerator}.')
    with st.expander('What do these terms mean?'):
        st.markdown('**log2 fold change (log2FC):** estimated expression difference on a log2 scale; +1 is approximately twofold higher, −1 twofold lower.  **p-value:** evidence against the null model under its assumptions.  **Adjusted p-value (FDR):** p-value adjusted for multiple tests (Benjamini–Hochberg).  **Candidates:** exploratory genes selected by a declared rule, not validated biomarkers.')
        st.caption('A gene may be statistically significant without being a causal driver or clinically useful marker.')
    scope = f'v25b_{fingerprint[:16]}'
    left, mid, right = st.columns(3)
    with left:
        strategy = st.selectbox('Exploratory candidate strategy',
            list(STRATEGY_EXPLANATIONS), key=f'{scope}_strategy',
            format_func=lambda x: STRATEGY_EXPLANATIONS[x][0],
            help='Choose how genes are highlighted for exploratory follow-up.')
    st.info(STRATEGY_EXPLANATIONS[strategy][1])
    with mid:
        n = st.number_input('N per direction' if strategy == 'balanced' else 'N genes (total)', min_value=1, max_value=100,
                            value=5, step=1, key=f'{scope}_n')
    with right:
        fdr = st.number_input('Exploratory FDR cutoff', min_value=0.001, max_value=1.0,
                              value=0.05, step=0.01, format='%.3f', key=f'{scope}_fdr')
    c1, c2 = st.columns(2)
    with c1:
        min_lfc = st.number_input('Minimum absolute log2FC for automatic selection',
            min_value=0.0, value=0.0, step=0.25, key=f'{scope}_min_lfc')
    with c2:
        p_column = st.selectbox('Volcano y-axis', ['pvalue', 'padj'],
            format_func=lambda x: '−log10(raw p-value)' if x == 'pvalue' else '−log10(adjusted p-value)',
            key=f'{scope}_p_column')
    manual = ()
    if strategy == 'manual':
        manual = tuple(st.multiselect('Select genes to highlight', sorted(results.index.tolist()),
                                     key=f'{scope}_manual'))
    payload = explorer_payload(results, strategy=strategy, n=int(n), fdr=float(fdr),
        min_abs_log2fc=float(min_lfc), manual_genes=manual, p_column=p_column)
    plot_table = payload['volcano'].loc[payload['volcano']['plot_eligible']].copy()
    plot_table.index.name = None
    counts = payload['volcano_audit']
    a, b, c = st.columns(3)
    a.metric('Genes analyzed', len(results))
    b.metric('FDR-significant genes', int(payload['volcano']['fdr_significant'].sum()))
    c.metric('Selected candidates', len(payload['selected_genes']))
    if counts['n_unplottable']:
        st.warning(f"{counts['n_unplottable']} genes could not be plotted due to unavailable values.")
    if counts['n_zero_p_capped']:
        st.info(f"{counts['n_zero_p_capped']} zero p-values were capped for display only; source values were unchanged.")
    if not plot_table.empty:
        fig = px.scatter(plot_table, x='log2FoldChange', y='volcano_y', color='category',
            hover_name='gene_id',
            hover_data={'baseMean': ':.3g', 'log2FoldChange': ':.3f',
                        'pvalue': ':.3g', 'padj': ':.3g', 'volcano_y': False,
                        'category': False} if 'baseMean' in plot_table else
                       {'log2FoldChange': ':.3f', 'pvalue': ':.3g', 'padj': ':.3g',
                        'volcano_y': False, 'category': False},
            color_discrete_map={'Higher in numerator':'#d45b54',
                                'Lower in numerator':'#4779bb',
                                'Not FDR significant':'#969ba5',
                                'FDR significant, zero effect':'#8465a8'},
            category_orders={'category':['Not FDR significant','Higher in numerator',
                                         'Lower in numerator','FDR significant, zero effect']})
        marked = plot_table.loc[plot_table['candidate_selected']]
        if not marked.empty:
            fig.add_trace(go.Scatter(x=marked['log2FoldChange'], y=marked['volcano_y'],
                mode='markers', name='Exploratory candidates',
                marker=candidate_marker_style(),
                text=marked['gene_id'], hovertemplate='%{text}<extra>Selected candidate</extra>'))
        fig.update_layout(xaxis_title='log2 fold change',
            yaxis_title='−log10(raw p-value)' if p_column == 'pvalue' else '−log10(adjusted p-value)',
            legend_title_text='Classification', height=490)
        st.plotly_chart(fig, width='stretch', key=f'{scope}_volcano')
    st.caption('Plot colors use adjusted p-values for FDR significance; the y-axis measure is independently selectable. Gold stars mark exploratory selections, not validated biomarkers.')
    st.write('**Selected exploratory candidates**')
    if payload['selected_genes']:
        st.dataframe(display_results_table(results.loc[payload['selected_genes']]),
                     width='stretch', hide_index=True)
    else:
        st.info('No genes satisfy this selection rule. Adjust display settings or select manually.')
    search = st.text_input('Find gene in the full DE results table', key=f'{scope}_search',
                           help='Case-insensitive substring search of gene identifiers.')
    shown = results.copy(deep=True)
    if search.strip():
        shown = shown.loc[shown.index.astype(str).str.contains(search.strip(), case=False, regex=False)]
    st.dataframe(display_results_table(shown), width='stretch', hide_index=True)
    with st.expander('Selection rule and volcano plotting audit'):
        st.json({'selection':payload['selection_audit'], 'volcano':payload['volcano_audit']})
    st.download_button('Download selected candidate IDs',
        data=('gene_id\n'+'\n'.join(payload['selected_genes'])+'\n').encode('utf-8'),
        file_name='omicsgpt_exploratory_candidates.csv', mime='text/csv', key=f'{scope}_download')
