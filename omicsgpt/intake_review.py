"""Serializable, non-mutating intake review summaries for a future Streamlit UI."""
from dataclasses import asdict
from typing import Any
import pandas as pd
from .intake import IntakeReport
from .design import DesignReport


def make_intake_review(counts: pd.DataFrame, metadata: pd.DataFrame,
                       intake: IntakeReport, design: DesignReport | None = None,
                       *, counts_sha256: str | None = None,
                       metadata_sha256: str | None = None) -> dict[str, Any]:
    """Generate preview and proposed actions; no data transformations occur."""
    return {
        'counts_sha256': counts_sha256,
        'metadata_sha256': metadata_sha256,
        'source_counts_shape': list(counts.shape),
        'source_metadata_shape': list(metadata.shape),
        'counts_preview': counts.head(5).iloc[:, :6].reset_index().astype(str).to_dict(orient='records'),
        'metadata_preview': metadata.head(5).astype(str).to_dict(orient='records'),
        'intake': asdict(intake),
        'design': asdict(design) if design is not None else None,
        'proposed_actions': [
            'Transpose counts to samples x genes' if intake.orientation == 'genes_by_samples' else
            'Keep samples x genes orientation' if intake.orientation == 'samples_by_genes' else
            'Resolve orientation with researcher',
            'Align samples to metadata order only after confirmation',
            'Do not modify, impute, round, or filter counts during intake',
        ],
        'requires_researcher_confirmation': True,
    }
