"""Audited, benchmark-only exclusion of unresolved duplicate TRNAV-CAC features.

Run as: python -m scripts.prepare_full_benchmark --counts ... --metadata ... --outdir ...
Never modifies source inputs. No model fitting or API calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


class BenchmarkPreparationError(ValueError):
    """Fail-closed source or policy validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(counts_path: Path, metadata_path: Path, outdir: Path, *,
            feature: str = 'TRNAV-CAC', expected_positions: tuple[int, ...] = (2031, 2032, 2152),
            min_count: int = 10, min_fraction: float = 0.20) -> dict:
    """Validate entire CSV, exclude only verified ambiguous features, and audit.

    Positions are zero-based CSV field indices including the first sample-ID field.
    """
    if type(min_count) is not int or min_count < 1:
        raise BenchmarkPreparationError('min_count must be a positive integer')
    if not isinstance(min_fraction, (float, int)) or not math.isfinite(min_fraction) or not 0 < min_fraction <= 1:
        raise BenchmarkPreparationError('min_fraction must be finite in (0,1]')
    if not expected_positions or any(type(x) is not int or x < 1 for x in expected_positions):
        raise BenchmarkPreparationError('Expected positions must be positive CSV field indices')
    if len(set(expected_positions)) != len(expected_positions):
        raise BenchmarkPreparationError('Expected positions must be unique')
    counts_path, metadata_path, outdir = map(lambda p: Path(p).resolve(), (counts_path, metadata_path, outdir))
    if not counts_path.is_file() or not metadata_path.is_file():
        raise BenchmarkPreparationError('Counts and metadata files must exist')
    if counts_path.parent == outdir or metadata_path.parent == outdir:
        raise BenchmarkPreparationError('Output directory must differ from source directory')
    if outdir.exists() and any(outdir.iterdir()):
        raise BenchmarkPreparationError('Output directory is not empty; choose a new location')
    # Metadata sample order is the reference; no silent sample reordering.
    with metadata_path.open('r', encoding='utf-8-sig', newline='') as handle:
        meta_reader = csv.DictReader(handle)
        if not meta_reader.fieldnames or 'Sample' not in meta_reader.fieldnames or len(set(meta_reader.fieldnames)) != len(meta_reader.fieldnames):
            raise BenchmarkPreparationError('Metadata requires unique headers including Sample')
        sample_ids = [row['Sample'] for row in meta_reader]
    if not sample_ids or any(not x or not x.strip() for x in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise BenchmarkPreparationError('Metadata sample identifiers must be nonblank and unique')
    min_samples = math.ceil(min_fraction * len(sample_ids))
    outdir.mkdir(parents=True, exist_ok=True)
    derived = outdir / 'counts_unique_features.csv'
    manifest_path = outdir / 'preparation_manifest.json'
    fd, temp_name = tempfile.mkstemp(prefix='.preparing_', suffix='.csv', dir=outdir)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with counts_path.open('r', encoding='utf-8-sig', newline='') as source, temp.open('w', encoding='utf-8', newline='') as target:
            reader = csv.reader(source, strict=True)
            writer = csv.writer(target, lineterminator='\n')
            header = next(reader, None)
            if not header or len(header) < 2 or header[0] != '':
                raise BenchmarkPreparationError('Expected an unnamed first sample-ID header')
            if any(not x.strip() for x in header[1:]):
                raise BenchmarkPreparationError('Unexpected blank gene header')
            actual_positions = tuple(i for i, name in enumerate(header) if name == feature)
            if actual_positions != expected_positions:
                raise BenchmarkPreparationError(f'{feature} positions differ: observed {actual_positions}, expected {expected_positions}')
            duplicate_names = {name: n for name, n in Counter(header[1:]).items() if n > 1}
            if duplicate_names != {feature: len(expected_positions)}:
                raise BenchmarkPreparationError(f'Unexpected duplicate gene headers: {duplicate_names}')
            exclude = set(expected_positions)
            retained_header = [value for i, value in enumerate(header) if i not in exclude]
            writer.writerow(retained_header)
            eligible_counts = [0] * (len(header) - 1)
            maxima = {i: 0 for i in expected_positions}
            rows = 0
            for rows, record in enumerate(reader, start=1):
                if len(record) != len(header):
                    raise BenchmarkPreparationError(f'Row {rows} has {len(record)} fields; expected {len(header)}')
                if rows > len(sample_ids) or record[0] != sample_ids[rows - 1]:
                    raise BenchmarkPreparationError(f'Sample mismatch or extra sample at row {rows}')
                if not record[0].strip():
                    raise BenchmarkPreparationError(f'Blank sample identifier at row {rows}')
                for pos, raw in enumerate(record[1:], start=1):
                    if not raw or not raw.isascii() or not raw.isdecimal():
                        raise BenchmarkPreparationError(f'Invalid nonnegative integer count at row {rows}, column {pos}')
                    value = int(raw)
                    if value >= min_count:
                        eligible_counts[pos - 1] += 1
                    if pos in exclude:
                        maxima[pos] = max(maxima[pos], value)
                writer.writerow([value for i, value in enumerate(record) if i not in exclude])
            if rows != len(sample_ids):
                raise BenchmarkPreparationError(f'Sample count mismatch: {rows} versus {len(sample_ids)}')
            failing_positions = [pos for pos in expected_positions if eligible_counts[pos - 1] < min_samples]
            if len(failing_positions) != len(expected_positions):
                raise BenchmarkPreparationError(f'Excluded features do not all fail filtering: {failing_positions}')
            retained_before = sum(n >= min_samples for n in eligible_counts)
            retained_after = sum(n >= min_samples for i, n in enumerate(eligible_counts, start=1) if i not in exclude)
            if retained_before != retained_after:
                raise BenchmarkPreparationError('Filtering eligibility changed after exclusion')
        temp.replace(derived)
        manifest = {
            'run_type': 'benchmark_only_duplicate_feature_exclusion_not_researcher_approval',
            'source_counts_sha256': sha256_file(counts_path),
            'source_metadata_sha256': sha256_file(metadata_path),
            'derived_counts_sha256': sha256_file(derived),
            'source_columns_including_sample_id': len(header),
            'samples': rows,
            'genes_before': len(header) - 1,
            'genes_after': len(retained_header) - 1,
            'excluded_feature_label': feature,
            'excluded_csv_field_positions_zero_based': list(expected_positions),
            'excluded_feature_maxima': {str(k): v for k, v in maxima.items()},
            'reason': 'Unresolved duplicate gene labels; all excluded features fail this benchmark filter',
            'filter': {'min_count': min_count, 'min_fraction': min_fraction,
                       'min_samples': min_samples, 'genes_eligible_before': retained_before,
                       'genes_eligible_after': retained_after},
            'raw_count_provenance': 'provisional_not_upstream_verified',
            'warning': 'This exclusion is valid only for this benchmark and filtering policy; not a general annotation-resolution rule.'
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        return manifest
    except Exception:
        temp.unlink(missing_ok=True)
        derived.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--min-count', type=int, default=10)
    parser.add_argument('--min-fraction', type=float, default=0.20)
    args = parser.parse_args()
    print(json.dumps(prepare(args.counts, args.metadata, args.outdir,
                             min_count=args.min_count, min_fraction=args.min_fraction), indent=2))


if __name__ == '__main__':
    main()
