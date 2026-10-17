"""Non-fitting full-transcriptome intake and library-size audit.

Run from repository root: python -m scripts.audit_full_counts --counts ... --metadata ... --outdir ...
No network access, model fitting, or mutation of input files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from omicsgpt.vnext_workflow import inspect_uploads, approve_review, prepare_approved_counts


_COUNT_PATTERN = re.compile(r"[0-9]+\Z")
_MAX_INT64 = np.iinfo(np.int64).max


def _validated_count_array(counts: pd.DataFrame) -> np.ndarray:
    """Return a validated integer array without mutating the supplied DataFrame.

    Accept integer dtype values and exact decimal integer strings. Reject float
    values (even 1.0), signs, whitespace, missing data and out-of-range values.
    """
    raw = counts.to_numpy(copy=False)
    if all(pd.api.types.is_integer_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
           for dtype in counts.dtypes):
        if any(pd.api.types.is_unsigned_integer_dtype(dtype) for dtype in counts.dtypes):
            if (raw > _MAX_INT64).any():
                raise ValueError('Counts exceed supported int64 range.')
        values = raw.astype(np.int64, copy=False)
        if (values < 0).any():
            raise ValueError('Negative counts are invalid.')
        return values

    # The strict importer intentionally retains CSV count fields as strings.
    # Check lexical form before conversion; never round or coerce floats.
    def parse_count(value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('Boolean counts are invalid.')
        if isinstance(value, (int, np.integer)):
            number = int(value)
        elif isinstance(value, str) and _COUNT_PATTERN.fullmatch(value):
            number = int(value)
        else:
            raise ValueError('Counts must be nonnegative integers or decimal integer strings.')
        if number < 0 or number > _MAX_INT64:
            raise ValueError('Counts must be nonnegative and within int64 range.')
        return number

    try:
        flat = np.fromiter((parse_count(value) for value in raw.flat),
                           dtype=np.int64, count=raw.size)
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError(f'Invalid count value in library-size audit: {exc}') from exc
    return flat.reshape(raw.shape)


def summarize_library_sizes(counts: pd.DataFrame, *, validated_values: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Compute per-sample count sums without normalizing or mutating counts."""
    if counts.empty or not counts.index.is_unique or not counts.columns.is_unique:
        raise ValueError('Counts must be nonempty with unique sample and gene IDs.')
    values = _validated_count_array(counts) if validated_values is None else validated_values
    if values.shape != counts.shape or values.dtype != np.dtype('int64'):
        raise ValueError('Validated numeric matrix has incompatible shape or dtype.')
    # Python integer accumulation avoids overflow in int64 sum.
    totals = [sum(map(int, row)) for row in values]
    if not all(v > 0 for v in totals):
        raise ValueError('At least one sample has zero total counts.')
    tab = pd.DataFrame({'sample_id': list(counts.index), 'library_size': totals})
    s = pd.Series(totals, dtype='float64')
    summary = {'min': int(min(totals)), 'median': float(s.median()),
               'mean': float(s.mean()), 'max': int(max(totals)),
               'max_to_min_ratio': float(max(totals) / min(totals)),
               'q1': float(s.quantile(.25)), 'q3': float(s.quantile(.75))}
    return tab, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--min-count', type=int, default=10)
    p.add_argument('--min-fraction', type=float, default=.20)
    args = p.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise SystemExit('Output directory is not empty; choose a new directory to preserve earlier audits.')
    if args.min_count < 0 or not math.isfinite(args.min_fraction) or not 0 < args.min_fraction <= 1:
        raise SystemExit('Invalid filtering policy: min-count >= 0 and 0 < min-fraction <= 1 required.')
    start = time.perf_counter()
    stages = {}
    counts_bytes = args.counts.read_bytes()
    metadata_bytes = args.metadata.read_bytes()
    stages['read_input_seconds'] = round(time.perf_counter() - start, 3)
    stage = time.perf_counter()
    bundle = inspect_uploads(counts_bytes, metadata_bytes,
        sample_id_col='Sample', group_col='condition',
        numerator='risk category: High', reference='risk category: Average',
        min_count=args.min_count, min_fraction=args.min_fraction)
    if bundle.config is None:
        raise RuntimeError(f'Intake blocked: {bundle.intake.issues}; design: {bundle.design.issues if bundle.design else None}')
    stages['inspect_seconds'] = round(time.perf_counter() - stage, 3)
    stage = time.perf_counter()
    # This approval is only a reproducible benchmark setup, not researcher authorization.
    fp = approve_review(bundle)
    matrix = prepare_approved_counts(bundle, fp, bundle.config)
    stages['prepare_seconds'] = round(time.perf_counter() - stage, 3)
    stage = time.perf_counter()
    # Validate and convert ONCE. All subsequent arithmetic uses the same numeric values.
    values = _validated_count_array(matrix)
    stages['numeric_validation_seconds'] = round(time.perf_counter() - stage, 3)
    stage = time.perf_counter()
    libraries, lib_summary = summarize_library_sizes(matrix, validated_values=values)
    stages['library_summary_seconds'] = round(time.perf_counter() - stage, 3)
    stage = time.perf_counter()
    min_samples = math.ceil(args.min_fraction * len(matrix))
    # Match the existing >= threshold policy, including min_count=0 semantics.
    keep = np.count_nonzero(values >= args.min_count, axis=0) >= min_samples
    all_zero = np.count_nonzero(~np.any(values, axis=0))
    stages['filter_summary_seconds'] = round(time.perf_counter() - stage, 3)
    group_counts = bundle.metadata.frame['condition'].value_counts().to_dict()
    report = {
        'run_type': 'noninteractive_intake_audit_no_statistical_fitting',
        'raw_count_status': 'provisional_unverified_upstream_provenance',
        'python_version': platform.python_version(), 'pandas_version': pd.__version__,
        'counts_sha256': hashlib.sha256(counts_bytes).hexdigest(),
        'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'configuration_fingerprint': fp, 'orientation': bundle.intake.orientation,
        'samples': int(matrix.shape[0]), 'genes_input': int(matrix.shape[1]),
        'genes_all_zero': int(all_zero),
        'filter': {'min_count': args.min_count, 'min_fraction': args.min_fraction,
                   'min_samples': min_samples, 'genes_retained': int(keep.sum()),
                   'genes_excluded': int((~keep).sum())},
        'group_sizes': {str(k): int(v) for k, v in group_counts.items()},
        'library_sizes': lib_summary,
        'stage_timings_seconds': stages,
        'elapsed_seconds': round(time.perf_counter() - start, 3),
        'note': 'Total counts and filtering are descriptive; integer counts alone do not establish raw-read provenance.'
    }
    # All computation completes before outputs are written. Use temporary files and
    # replace them only after both have been successfully generated.
    args.outdir.mkdir(parents=True, exist_ok=True)
    temp_csv = args.outdir / 'sample_library_sizes.csv.tmp'
    temp_json = args.outdir / 'intake_audit.json.tmp'
    try:
        libraries.to_csv(temp_csv, index=False)
        temp_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp_csv.replace(args.outdir / 'sample_library_sizes.csv')
        temp_json.replace(args.outdir / 'intake_audit.json')
    finally:
        temp_csv.unlink(missing_ok=True)
        temp_json.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
