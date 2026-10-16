"""OmicsGPT vNext: validated bulk RNA-seq intake and approved PyDESeq2."""
import streamlit as st
import time
import hashlib
import json
from datetime import datetime, timezone
from omicsgpt.run_state import (reconcile_state, record_approval, is_approved,
    record_completed, current_result, clear_result)
from omicsgpt.intake_suggestions import suggest_columns, compact_group_sizes

_render_started = time.perf_counter()
from omicsgpt.vnext_workflow import (
    inspect_uploads, approve_review, prepare_approved_counts, fingerprint_config
)

st.set_page_config(page_title='OmicsGPT vNext', layout='wide')
st.title('OmicsGPT vNext')
st.caption('V2.5B — Researcher-approved PyDESeq2 and exploratory results explorer. No AI calls.')

st.selectbox('Data modality', ['Bulk RNA-seq raw counts'], help='Other omics modalities are not supported in V1.')
counts_upload = st.file_uploader('Counts CSV', type='csv', key='vnext_counts')
metadata_upload = st.file_uploader('Metadata CSV', type='csv', key='vnext_metadata')

if not counts_upload or not metadata_upload:
    reconcile_state(st.session_state, None)
    st.info('Upload both CSV files to inspect their structure and design.')
    st.stop()

# Strictly read the header only for user-selectable metadata fields; the full reader
# runs in inspect_uploads and blocks malformed files before confirmation.
from omicsgpt.csv_import import read_strict_csv, CSVImportError
try:
    meta_source = read_strict_csv(metadata_upload.getvalue())
except (CSVImportError, ValueError, UnicodeError) as exc:
    reconcile_state(st.session_state, None)
    st.error(f'Metadata import blocked: {exc}')
    st.stop()

# Read the count IDs only to propose defaults; full validation still happens below.
try:
    count_source = read_strict_csv(counts_upload.getvalue(), use_identifier_as_index=True)
except (CSVImportError, ValueError, UnicodeError) as exc:
    reconcile_state(st.session_state, None)
    st.error(f'Counts import blocked: {exc}')
    st.stop()

suggestions = suggest_columns(count_source.frame, meta_source.frame)
fields = list(meta_source.frame.columns)
# Widget identity changes only when uploaded bytes change, never when another
# selectbox changes. Thus deliberate user overrides survive ordinary reruns.
source_key = hashlib.sha256(counts_upload.getvalue() + metadata_upload.getvalue()).hexdigest()[:16]
st.caption('Suggested defaults are based on exact identifier matching and repeated categories. Confirm both selections.')
if suggestions.sample_id_ambiguous:
    st.warning('Multiple metadata columns match sample identifiers. Please verify the sample-ID selection.')
if suggestions.sample_id_match_count == 0:
    st.warning('No metadata column uniquely matches all count sample IDs. Select the correct ID field.')
sample_id_col = st.selectbox('Metadata sample ID', fields,
                             index=fields.index(suggestions.sample_id_column),
                             key=f'vnext_sample_{source_key}', help=suggestions.sample_id_reason)
condition_options = [f for f in fields if f != sample_id_col]
if not condition_options:
    reconcile_state(st.session_state, None)
    st.error('Metadata must include a grouping variable in addition to sample IDs.')
    st.stop()
preferred_group = suggestions.grouping_column if suggestions.grouping_column in condition_options else condition_options[0]
group_col = st.selectbox('Grouping variable', condition_options,
                         index=condition_options.index(preferred_group),
                         key=f'vnext_group_{source_key}_{sample_id_col}',
                         help='Researcher must verify that this represents the intended experimental groups.')
levels = sorted(meta_source.frame[group_col].astype(str).unique().tolist())
if len(levels) < 2:
    reconcile_state(st.session_state, None)
    st.error('At least two grouping levels are required.')
    st.stop()
if len(levels) > 20:
    st.warning(f'{len(levels)} unique values detected in the grouping variable. '
               'This may be an identifier rather than a biological condition.')
numerator = st.selectbox('Test / numerator', levels, index=1,
                         key=f'vnext_numerator_{source_key}_{sample_id_col}_{group_col}')
reference = st.selectbox('Reference / denominator', levels, index=0,
                         key=f'vnext_reference_{source_key}_{sample_id_col}_{group_col}')
covariates = st.multiselect('Categorical covariates (optional)',
                            [f for f in condition_options if f != group_col],
                            key=f'vnext_covariates_{source_key}_{sample_id_col}_{group_col}')

st.subheader('Statistical settings')
min_count = st.number_input('Minimum counts per qualifying sample', min_value=0,
                            max_value=1000000, value=10, step=1,
                            help='Gene must have at least this many counts in the required fraction of samples.')
min_fraction_percent = st.number_input('Minimum percentage of samples', min_value=1,
                                       max_value=100, value=20, step=1,
                                       help='Required sample count is rounded up.')
st.caption('Engine: PyDESeq2; requested dispersion trend: parametric (automatic internal fallback recorded); Cook outlier refitting: enabled.')

validation_started = time.perf_counter()
try:
    bundle = inspect_uploads(counts_upload.getvalue(), metadata_upload.getvalue(),
                             sample_id_col=sample_id_col, group_col=group_col,
                             numerator=numerator, reference=reference,
                             covariates=tuple(covariates), min_count=int(min_count),
                             min_fraction=float(min_fraction_percent) / 100.0)
except (CSVImportError, ValueError, TypeError, UnicodeError) as exc:
    reconcile_state(st.session_state, None)
    st.error(f'Intake blocked: {exc}')
    st.stop()

validation_seconds = time.perf_counter() - validation_started
left, right = st.columns(2)
with left:
    st.subheader('Source counts')
    st.write(f'{bundle.counts.frame.shape[0]:,} rows × {bundle.counts.frame.shape[1]:,} columns')
    st.dataframe(bundle.counts.frame.iloc[:5, :6])
with right:
    st.subheader('Source metadata')
    st.write(f'{len(bundle.metadata.frame):,} records')
    st.dataframe(bundle.metadata.frame.head())

st.subheader('Data Intake Review')
valid = bundle.config is not None
if valid:
    st.success('Intake and experimental design are ready for researcher confirmation.')
else:
    st.warning('Review the issues below before confirmation. No analysis will run.')
c1, c2, c3 = st.columns(3)
c1.metric('Metadata samples', bundle.intake.n_metadata_samples)
c2.metric('Genes (interpreted)', bundle.counts.frame.shape[1] if bundle.intake.orientation == 'samples_by_genes' else bundle.counts.frame.shape[0] if bundle.intake.orientation == 'genes_by_samples' else 'Unresolved')
c3.metric('Matched sample IDs', max(bundle.intake.matched_row_ids, bundle.intake.matched_column_ids))
st.write(f'**Orientation:** {bundle.intake.orientation or "unresolved"}')
for warning in bundle.intake.warnings:
    st.warning(warning)
for issue in bundle.intake.issues:
    st.error(issue)
if bundle.design is not None:
    st.write('**Experimental groups**')
    group_rows, hidden_groups = compact_group_sizes(bundle.design.group_sizes)
    st.dataframe(group_rows, hide_index=True, width='stretch')
    if hidden_groups:
        st.caption(f'{hidden_groups} additional groups not displayed. Choose a grouping variable with repeated biological categories.')
    st.write(f'**Design rank:** {bundle.design.design_rank} / {bundle.design.design_columns}')
    for warning in bundle.design.warnings:
        st.warning(warning)
    for issue in bundle.design.issues:
        st.error(issue)
with st.expander('Development timing (server-side only)'):
    st.write(f'Intake inspection: {validation_seconds:.3f} seconds')
    st.write(f'Script execution through this panel: {time.perf_counter() - _render_started:.3f} seconds')
    st.caption('These measurements exclude Docker startup, browser rendering, and network transfer.')

if bundle.config is None:
    reconcile_state(st.session_state, None)
    st.error('Resolve all validation blockers before confirmation.')
    st.stop()

st.subheader('Proposed analysis contract')
st.write(f'**Contrast:** {numerator} versus {reference}; positive log2FC will mean higher in {numerator}.')
st.write('**Proposed transformations:** ' + ', '.join(bundle.config['transformations']))
st.write('**Counts:** No imputation, rounding, or filtering at intake.')
st.write(f'**Statistical engine:** PyDESeq2; gene filter: ≥{int(min_count)} counts in ≥{int(min_fraction_percent)}% of samples; parametric requested; Cook refitting enabled.')
with st.expander('Source hashes and complete configuration'):
    st.json(bundle.config)

current_fp = fingerprint_config(bundle.config)
reconcile_state(st.session_state, current_fp)

if st.button('Confirm this exact dataset, design, and statistical settings', type='primary'):
    record_approval(st.session_state, approve_review(bundle))

if is_approved(st.session_state, current_fp):
    st.success('Researcher approval matches the current scientific configuration.')
    try:
        prepared = prepare_approved_counts(bundle, current_fp, bundle.config)
    except (PermissionError, ValueError) as exc:
        reconcile_state(st.session_state, None)
        st.error(f'Approved data preparation failed: {exc}')
        st.stop()
    st.write(f'Prepared matrix: {prepared.shape[0]:,} samples × {prepared.shape[1]:,} genes')
    with st.expander('Prepared matrix preview'):
        st.dataframe(prepared.iloc[:5, :6])
    if st.button('Run approved PyDESeq2 analysis', type='primary'):
        clear_result(st.session_state)
        try:
            # Lazy import: intake does not initialize the statistical stack.
            from omicsgpt.guarded_execution import execute_approved_analysis
            with st.spinner('Fitting PyDESeq2 to the approved data...'):
                outcome = execute_approved_analysis(bundle, current_fp, bundle.config, n_cpus=1)
            record_completed(st.session_state, current_fp, outcome)
        except Exception as exc:
            clear_result(st.session_state)
            st.error(f'Analysis failed; no results were saved: {type(exc).__name__}: {exc}')
else:
    st.info('Review and confirm this configuration before running statistical analysis.')

outcome = current_result(st.session_state, current_fp)
if outcome is not None:
    st.subheader('Completed differential expression')
    table = outcome['results'].copy()
    st.success(f'PyDESeq2 completed for {len(table):,} genes.')
    st.dataframe(table.reset_index().rename(columns={'index': 'gene_id'}), width='stretch')
    # Display-only exploration: does not change approval or rerun PyDESeq2.
    from omicsgpt.results_panel import render_results_panel
    render_results_panel(table, numerator=numerator, reference=reference, fingerprint=current_fp)
    st.subheader('Filtering and model diagnostics')
    st.json({'filter_audit': outcome['filter_audit'], 'diagnostics': outcome['diagnostics'],
             'model': outcome['model']})
    st.download_button('Download differential expression CSV',
                       data=table.to_csv(index_label='gene_id').encode('utf-8'),
                       file_name='omicsgpt_differential_expression.csv', mime='text/csv')
    manifest = {'scientific_configuration': bundle.config, 'configuration_fingerprint': current_fp,
                'filter_audit': outcome['filter_audit'], 'diagnostics': outcome['diagnostics'],
                'model': outcome['model'], 'results_rows': len(table)}
    st.download_button('Download analysis manifest JSON',
                       data=json.dumps(manifest, indent=2, default=str).encode('utf-8'),
                       file_name='omicsgpt_run_manifest.json', mime='application/json')
