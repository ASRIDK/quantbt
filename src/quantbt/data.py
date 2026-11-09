"""Data loaders returning a clean, date-indexed OHLCV DataFrame."""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def load_csv(path, date_column: str = "date") -> pd.DataFrame:
    """Load OHLCV bars from a CSV file.

    Column names are matched case-insensitively. The result is indexed by a
    sorted, unique ``DatetimeIndex`` with lowercase float columns
    ``open, high, low, close, volume``.

    Raises ``ValueError`` if a required column is missing or a date cannot
    be parsed.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_column = date_column.lower()
    if date_column not in df.columns:
        raise ValueError(f"missing date column: {date_column!r}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df[date_column] = pd.to_datetime(df[date_column], errors="raise")
    df = df.set_index(date_column).sort_index()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")]
    return df[list(REQUIRED_COLUMNS)].astype(float)
