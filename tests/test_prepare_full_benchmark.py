"""Synthetic fail-closed tests for benchmark-only duplicate-feature handling."""
import csv
from pathlib import Path

import pytest

from scripts.prepare_full_benchmark import prepare, BenchmarkPreparationError, sha256_file


def fixture(tmp_path: Path, *, duplicate_values=('1','2','0'), bad=None):
    counts = tmp_path / 'source.csv'
    meta = tmp_path / 'meta.csv'
    with counts.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'G1', 'TRNAV-CAC', 'TRNAV-CAC', 'G2', 'TRNAV-CAC'])
        for i in range(5):
            row = [f'S{i}', '20', *duplicate_values[:2], '30', duplicate_values[2]]
            if bad and i == 2:
                row[bad[0]] = bad[1]
            writer.writerow(row)
    with meta.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample', 'condition'])
        for i in range(5):
            writer.writerow([f'S{i}', 'A' if i % 2 else 'B'])
    return counts, meta


def run(counts, meta, out, **kwargs):
    return prepare(counts, meta, out, expected_positions=(2,3,5), **kwargs)


def test_success_preserves_source_and_filter(tmp_path):
    counts, meta = fixture(tmp_path)
    before = sha256_file(counts)
    report = run(counts, meta, tmp_path/'out')
    assert sha256_file(counts) == before
    assert report['genes_before'] == 5 and report['genes_after'] == 2
    assert report['filter']['genes_eligible_before'] == report['filter']['genes_eligible_after'] == 2
    assert report['excluded_csv_field_positions_zero_based'] == [2,3,5]
    assert report['source_counts_sha256'] == before
    assert (tmp_path/'out'/'counts_unique_features.csv').exists()


def test_reject_if_duplicate_passes_filter(tmp_path):
    counts, meta = fixture(tmp_path, duplicate_values=('20','2','0'))
    with pytest.raises(BenchmarkPreparationError, match='fail filtering'):
        run(counts, meta, tmp_path/'out')
    assert not (tmp_path/'out'/'counts_unique_features.csv').exists()


@pytest.mark.parametrize('bad', [(2,'1.5'), (4,'-1'), (1,'')])
def test_reject_invalid_count(tmp_path, bad):
    counts, meta = fixture(tmp_path, bad=bad)
    with pytest.raises(BenchmarkPreparationError, match='Invalid nonnegative'):
        run(counts, meta, tmp_path/'out')


def test_reject_unexpected_positions(tmp_path):
    counts, meta = fixture(tmp_path)
    with pytest.raises(BenchmarkPreparationError, match='positions differ'):
        prepare(counts, meta, tmp_path/'out')


def test_reject_existing_output(tmp_path):
    counts, meta = fixture(tmp_path)
    out = tmp_path/'out'
    out.mkdir()
    (out/'keep.txt').write_text('existing')
    with pytest.raises(BenchmarkPreparationError, match='not empty'):
        run(counts, meta, out)
    assert (out/'keep.txt').read_text() == 'existing'


def test_reject_metadata_order_mismatch(tmp_path):
    counts, meta = fixture(tmp_path)
    text = meta.read_text().replace('S0,B', 'Z0,B')
    meta.write_text(text)
    with pytest.raises(BenchmarkPreparationError, match='Sample mismatch'):
        run(counts, meta, tmp_path/'out')
