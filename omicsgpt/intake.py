"""Non-mutating bulk RNA-seq count inspection (no statistical estimation)."""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd

Status = Literal['ready_for_confirmation', 'needs_user_resolution', 'blocked']


@dataclass(frozen=True)
class IntakeReport:
    status: Status
    orientation: str | None
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    n_metadata_samples: int
    n_count_rows: int
    n_count_columns: int
    matched_row_ids: int
    matched_column_ids: int
    all_zero_genes: int | None


def inspect_bulk_counts(counts: pd.DataFrame, metadata: pd.DataFrame, sample_id_col: str) -> IntakeReport:
    """Inspect uploaded count matrix without modifying or repairing source data.

    This does NOT approve a design, infer a contrast, or start computation.
    The caller must obtain researcher confirmation before preparation.
    """
    issues: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    resolution: list[str] = []
    if not isinstance(counts, pd.DataFrame) or not isinstance(metadata, pd.DataFrame):
        raise TypeError('counts and metadata must be pandas DataFrames')
    if counts.empty or metadata.empty:
        blockers.append('Count matrix and metadata must be nonempty.')
    if sample_id_col not in metadata.columns:
        blockers.append(f'Metadata sample ID column {sample_id_col!r} is missing.')
        sample_ids = pd.Index([])
    else:
        sample_ids = pd.Index(metadata[sample_id_col])
        if sample_ids.isna().any() or (sample_ids.astype(str).str.strip() == '').any():
            blockers.append('Metadata contains missing/blank sample identifiers.')
        if sample_ids.has_duplicates:
            blockers.append('Metadata contains duplicate sample identifiers.')
    if counts.index.has_duplicates:
        blockers.append('Count matrix has duplicate row identifiers.')
    if counts.columns.has_duplicates:
        blockers.append('Count matrix has duplicate column identifiers.')
    if counts.index.isna().any() or counts.columns.isna().any():
        blockers.append('Count matrix has missing row or column identifiers.')
    row_matches = len(sample_ids.intersection(counts.index))
    col_matches = len(sample_ids.intersection(counts.columns))
    complete_rows = len(sample_ids) > 0 and row_matches == len(sample_ids)
    complete_cols = len(sample_ids) > 0 and col_matches == len(sample_ids)
    orientation = None
    if complete_rows and not complete_cols:
        orientation = 'samples_by_genes'
        if len(counts.index) != len(sample_ids):
            resolution.append('Counts include additional samples not in metadata; explicit selection required.')
    elif complete_cols and not complete_rows:
        orientation = 'genes_by_samples'
        if len(counts.columns) != len(sample_ids):
            resolution.append('Counts include additional samples not in metadata; explicit selection required.')
    elif complete_rows and complete_cols:
        resolution.append('Both dimensions match metadata sample identifiers; orientation is ambiguous.')
    else:
        resolution.append(f'Sample IDs do not fully match exactly one axis (rows={row_matches}, columns={col_matches}, metadata={len(sample_ids)}).')

    # Strict numeric inspection: never convert or round the original frame.
    invalid = []
    for row_id, row in counts.iterrows():
        for col_id, value in row.items():
            if pd.isna(value):
                invalid.append((row_id, col_id, 'missing'))
                continue
            if isinstance(value, (bool, np.bool_)):
                invalid.append((row_id, col_id, 'boolean'))
                continue
            try:
                numeric = float(value)
            except (ValueError, TypeError, OverflowError):
                invalid.append((row_id, col_id, 'non-numeric'))
                continue
            if not np.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
                invalid.append((row_id, col_id, 'non-finite, negative, or fractional'))
    if invalid:
        examples = ', '.join(f'({r!r}, {c!r}: {reason})' for r, c, reason in invalid[:3])
        blockers.append(f'{len(invalid)} invalid count values; examples: {examples}.')
    zero_genes = None
    if not invalid and orientation:
        numeric_counts = counts.astype(float)
        zero_genes = int(((numeric_counts.sum(axis=1) if orientation == 'genes_by_samples' else numeric_counts.sum(axis=0)) == 0).sum())
        if zero_genes:
            warnings.append(f'{zero_genes} all-zero genes detected; no filtering performed during inspection.')
    issues.extend(blockers)
    issues.extend(resolution)
    status: Status = 'blocked' if blockers else 'needs_user_resolution' if resolution else 'ready_for_confirmation'
    return IntakeReport(status, orientation, tuple(issues), tuple(warnings), len(sample_ids), len(counts.index), len(counts.columns), row_matches, col_matches, zero_genes)


def prepare_confirmed_counts(counts: pd.DataFrame, metadata: pd.DataFrame, sample_id_col: str, confirmed_orientation: str) -> pd.DataFrame:
    """Return a NEW samples-by-genes matrix after explicit orientation confirmation.

    No filtering, dropping, imputation, rounding or gene-ID conversion occurs.
    """
    report = inspect_bulk_counts(counts, metadata, sample_id_col)
    if report.status != 'ready_for_confirmation':
        raise ValueError(f'Intake not ready: {report.issues}')
    if confirmed_orientation != report.orientation:
        raise ValueError('Confirmed orientation does not match inspected orientation.')
    sample_ids = list(metadata[sample_id_col])
    matrix = counts.T if confirmed_orientation == 'genes_by_samples' else counts
    return matrix.loc[sample_ids].copy(deep=True)
