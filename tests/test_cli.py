import json

import numpy as np
import pandas as pd
import pytest

from quantbt.cli import main


@pytest.fixture
def csv_path(tmp_path):
    n = 60
    close = 100 + np.cumsum(np.sin(np.arange(n) / 4.0))
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        }
    )
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_run_prints_table(csv_path, capsys):
    assert main(["run", csv_path, "--quantity", "10"]) == 0
    out = capsys.readouterr().out
    assert "sharpe" in out and "max_drawdown" in out


def test_run_json_sma(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "sma", "--fast", "3", "--slow", "8", "--json"]
    assert main(argv + ["--commission-bps", "5"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert {"sharpe", "sortino", "var", "cvar", "calmar"} <= set(stats)


def test_missing_file_returns_error(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.csv")]) == 1
    assert "error" in capsys.readouterr().err


def test_bad_sma_windows_returns_error(csv_path, capsys):
    assert main(["run", csv_path, "--strategy", "sma", "--fast", "9", "--slow", "3"]) == 1
    assert "fast < slow" in capsys.readouterr().err
