import pandas as pd
import pytest

from quantbt.data import load_csv, load_yfinance


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


def _fake_yf(frame):
    import types

    mod = types.ModuleType("yfinance")
    mod.calls = []

    def download(symbol, **kwargs):
        mod.calls.append((symbol, kwargs))
        return frame

    mod.download = download
    return mod


def test_load_yfinance_normalises_multiindex_and_tz(monkeypatch):
    import sys

    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-02"], tz="America/New_York")
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["SPY"]]
    )
    frame = pd.DataFrame([[3, 4, 2, 3.5, 30], [1, 2, 0.5, 1.5, 10]], index=idx, columns=cols)
    fake = _fake_yf(frame)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    df = load_yfinance("SPY", start="2024-01-01")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is None and df.index.is_monotonic_increasing
    assert fake.calls[0][0] == "SPY" and fake.calls[0][1]["start"] == "2024-01-01"


def test_load_yfinance_empty(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf(pd.DataFrame()))
    with pytest.raises(ValueError, match="no data"):
        load_yfinance("NOPE")


def test_load_yfinance_missing_package(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match="quantbt\\[yfinance\\]"):
        load_yfinance("SPY")
