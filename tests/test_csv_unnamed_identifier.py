"""Regression checks for explicitly interpreted unnamed CSV row identifiers."""
import hashlib
import pandas as pd
import pytest

from omicsgpt.csv_import import CSVImportError, read_strict_csv


def test_unnamed_first_column_is_index_only_when_requested():
    raw = b',DDX11L1,WASH7P\nGSM001,6,303\nGSM002,0,12\n'
    result = read_strict_csv(raw, use_identifier_as_index=True)
    assert result.frame.shape == (2, 2)
    assert result.frame.index.tolist() == ['GSM001', 'GSM002']
    assert result.frame.columns.tolist() == ['DDX11L1', 'WASH7P']
    assert result.frame.loc['GSM001', 'WASH7P'] == '303'
    assert result.sha256 == hashlib.sha256(raw).hexdigest()
    assert result.identifier_column == ''
    with pytest.raises(CSVImportError, match='Blank CSV column header'):
        read_strict_csv(raw)


@pytest.mark.parametrize('raw', [
    b'id,,A\nx,1,2\n',
    b',A,\nx,1,2\n',
    b' ,A\nx,1\n',
    b',A,A\nx,1,2\n',
    b',A\nx,1\nx,2\n',
    b',A\n,1\n',
    b',A\nx,1,2\n',
])
def test_bad_unnamed_identifier_inputs_still_block(raw):
    with pytest.raises(CSVImportError):
        read_strict_csv(raw, use_identifier_as_index=True)


def test_unnamed_identifier_can_be_gene_axis():
    raw = b',S01,S02\nENSG00000012048.24,1,3\nLOC100996442,0,4\n'
    parsed = read_strict_csv(raw, use_identifier_as_index=True)
    assert parsed.frame.index.tolist() == ['ENSG00000012048.24', 'LOC100996442']
    assert parsed.frame.columns.tolist() == ['S01', 'S02']


def test_source_is_not_modified():
    raw = b',GeneA\n001,4\n002,0\n'
    original = bytes(raw)
    parsed = read_strict_csv(raw, use_identifier_as_index=True)
    assert raw == original
    assert parsed.frame.index.tolist() == ['001', '002']
