"""Read-only Streamlit preview; NOT an authorization gate for computation."""
import streamlit as st
from .csv_import import CSVImportError, read_strict_csv
from .intake import inspect_bulk_counts
from .intake_review import make_intake_review


def render_read_only_intake(counts_upload, metadata_upload):
    """Inspect upload bytes without consuming file pointers or changing session analysis state."""
    st.caption('Intake preview (experimental): inspection only; legacy analysis remains unchanged.')
    if counts_upload is None or metadata_upload is None:
        st.info('Upload both counts and metadata to inspect them.')
        return
    try:
        counts = read_strict_csv(counts_upload.getvalue(), use_identifier_as_index=True)
        metadata = read_strict_csv(metadata_upload.getvalue())
    except (CSVImportError, OSError, UnicodeError) as exc:
        st.error(f'CSV import blocked: {exc}')
        return

    sample_col = st.selectbox('Metadata sample-ID column', list(metadata.frame.columns),
                              key='intake_preview_sample_id')
    report = inspect_bulk_counts(counts.frame, metadata.frame, sample_col)
    review = make_intake_review(counts.frame, metadata.frame, report,
                                counts_sha256=counts.sha256, metadata_sha256=metadata.sha256)
    st.write(f"**Intake status:** `{report.status}`")
    st.write(f"Counts: {counts.frame.shape[0]:,} rows x {counts.frame.shape[1]:,} columns; "
             f"metadata: {len(metadata.frame):,} records")
    st.write(f"Orientation: `{report.orientation or 'unresolved'}`; "
             f"sample-ID matches: rows {report.matched_row_ids}, columns {report.matched_column_ids} "
             f"of {report.n_metadata_samples}")
    if report.issues:
        for issue in report.issues:
            st.error(issue)
    if report.warnings:
        for warning in report.warnings:
            st.warning(warning)
    st.write('**Source counts preview** (first 5 rows, 6 columns)')
    st.dataframe(counts.frame.iloc[:5, :6], use_container_width=True)
    st.write('**Source metadata preview** (first 5 rows)')
    st.dataframe(metadata.frame.head(5), use_container_width=True)
    st.write('**Proposed actions (not executed)**')
    for action in review['proposed_actions']:
        st.write(f'- {action}')
    with st.expander('Source SHA-256 hashes'):
        st.code(f"Counts: {counts.sha256}\nMetadata: {metadata.sha256}")
    st.warning('This panel does NOT gate the existing Apply Configuration button. '
               'Do not interpret a successful preview as approval to run the legacy analysis.')
