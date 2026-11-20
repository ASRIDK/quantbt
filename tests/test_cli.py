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


def test_json_output_is_strict_when_profit_factor_infinite(csv_path, capsys):
    # Buy-and-hold on a rising series has gains but no losing periods.
    import pandas as pd

    rising = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "open": range(100, 110),
            "high": range(101, 111),
            "low": range(99, 109),
            "close": range(100, 110),
            "volume": 1000,
        }
    )
    path = csv_path.replace("prices.csv", "rising.csv")
    rising.to_csv(path, index=False)
    assert main(["run", path, "--quantity", "10", "--json"]) == 0
    out = capsys.readouterr().out
    assert "Infinity" not in out
    assert json.loads(out)["profit_factor"] is None


def test_run_json_zscore(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "zscore", "--window", "8", "--entry", "0.5", "--json"]
    assert main(argv) == 0
    assert "sharpe" in json.loads(capsys.readouterr().out)


def test_bad_zscore_params_return_error(csv_path, capsys):
    assert main(["run", csv_path, "--strategy", "zscore", "--entry", "0"]) == 1
    assert "entry" in capsys.readouterr().err


def test_run_json_rsi(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "rsi", "--rsi-window", "5", "--json"]
    assert main(argv) == 0
    assert "sharpe" in json.loads(capsys.readouterr().out)


def test_bad_rsi_params_return_error(csv_path, capsys):
    assert main(["run", csv_path, "--strategy", "rsi", "--oversold", "80"]) == 1
    assert "oversold" in capsys.readouterr().err


def test_run_json_donchian(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "donchian", "--entry-window", "10", "--json"]
    assert main(argv) == 0
    assert "sharpe" in json.loads(capsys.readouterr().out)


def test_bad_donchian_params_return_error(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "donchian", "--entry-window", "5", "--exit-window", "9"]
    assert main(argv) == 1
    assert "exit" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "quantbt" in capsys.readouterr().out


def test_equity_csv_export(csv_path, tmp_path, capsys):
    out = tmp_path / "equity.csv"
    assert main(["run", csv_path, "--equity-csv", str(out)]) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["date", "equity"]
    assert len(df) == 60


def test_equity_csv_bad_path_returns_error(csv_path, tmp_path, capsys):
    bad = tmp_path / "missing" / "equity.csv"
    assert main(["run", csv_path, "--equity-csv", str(bad)]) == 1
    assert "error" in capsys.readouterr().err
