import pandas as pd
import pytest

from quantbt.data import load_csv


def _write(tmp_path, text):
    p = tmp_path / "bars.csv"
    p.write_text(text)
    return p


def test_load_csv_sorts_dedups_and_normalises(tmp_path):
    p = _write(
        tmp_path,
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-03,3,4,2,3.5,30\n"
        "2024-01-02,1,2,0.5,1.5,10\n"
        "2024-01-02,1,2,0.5,1.7,11\n",
    )
    df = load_csv(p)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert len(df) == 2
    assert df.loc["2024-01-02", "close"] == 1.7  # last duplicate wins


def test_load_csv_missing_columns(tmp_path):
    p = _write(tmp_path, "date,open,close\n2024-01-02,1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_csv(p)


def test_load_csv_missing_date(tmp_path):
    p = _write(tmp_path, "open,high,low,close,volume\n1,2,0,1,5\n")
    with pytest.raises(ValueError, match="date column"):
        load_csv(p)
