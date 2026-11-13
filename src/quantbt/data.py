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
    return _normalise(df.set_index(date_column))


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate and select OHLCV columns from a date-indexed frame."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize(None) if idx.tz is not None else idx
    df = df.sort_index()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")]
    return df[list(REQUIRED_COLUMNS)].astype(float)


def load_yfinance(symbol: str, start=None, end=None, interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV bars for ``symbol`` from Yahoo Finance.

    Requires the optional ``yfinance`` package (``pip install quantbt[yfinance]``).
    Returns the same clean, date-indexed OHLCV frame as :func:`load_csv`.

    Raises ``ImportError`` if yfinance is not installed and ``ValueError`` if
    no data comes back or required columns are missing.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "yfinance is required for load_yfinance; install with "
            "'pip install quantbt[yfinance]'"
        ) from exc

    raw = yf.download(
        symbol, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False,
    )
    if raw is None or len(raw) == 0:
        raise ValueError(f"no data returned for symbol {symbol!r}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return _normalise(raw)
