"""Strict, non-repairing CSV loading for OmicsGPT bulk-count intake."""
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
import csv
import hashlib
import pandas as pd


class CSVImportError(ValueError):
    """Input cannot be interpreted without an explicit researcher correction."""


@dataclass(frozen=True)
class ImportedCSV:
    frame: pd.DataFrame
    sha256: str
    n_rows: int
    n_columns: int
    identifier_column: str


def read_strict_csv(source: str | Path | bytes, *, identifier_column: str | None = None,
                    use_identifier_as_index: bool = False) -> ImportedCSV:
    """Read CSV without header mangling, type inference, or identifier coercion.

    CSV is decoded as UTF-8 with optional BOM; all values remain strings.
    Callers validate count values separately. The first column is the default ID.
    """
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise CSVImportError('CSV must use UTF-8 encoding; no encoding substitution was attempted.') from exc
    try:
        records = list(csv.reader(text.splitlines(keepends=True), strict=True))
    except csv.Error as exc:
        raise CSVImportError(f'Malformed CSV: {exc}') from exc
    if len(records) < 2 or not records[0]:
        raise CSVImportError('CSV must contain a header and at least one data row.')
    header = records[0]
    if any(not x.strip() for x in header):
        raise CSVImportError('Blank CSV column header detected.')
    if len(set(header)) != len(header):
        raise CSVImportError('Duplicate CSV column headers detected before pandas import.')
    if any(len(row) != len(header) for row in records[1:]):
        raise CSVImportError('CSV contains a row with a different number of fields.')
    if any(not row or all(not cell.strip() for cell in row) for row in records[1:]):
        raise CSVImportError('CSV contains an empty data row.')
    ident = identifier_column if identifier_column is not None else header[0]
    if ident not in header:
        raise CSVImportError(f'Identifier column {ident!r} is absent.')
    frame = pd.DataFrame(records[1:], columns=header, dtype=object)
    ids = frame[ident]
    if ids.eq('').any() or ids.str.strip().eq('').any():
        raise CSVImportError('Blank identifiers detected.')
    if ids.duplicated().any():
        raise CSVImportError('Duplicate identifiers detected.')
    if use_identifier_as_index:
        frame = frame.set_index(ident, drop=True)
    return ImportedCSV(frame=frame, sha256=digest, n_rows=len(records)-1,
                       n_columns=len(header), identifier_column=ident)
