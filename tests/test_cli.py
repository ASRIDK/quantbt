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


def test_run_json_ema(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "ema", "--ema-fast", "3", "--ema-slow", "8", "--json"]
    assert main(argv) == 0
    assert "sharpe" in capsys.readouterr().out


def test_bad_ema_params_return_error(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "ema", "--ema-fast", "9", "--ema-slow", "9"]
    assert main(argv) == 1
    assert "error" in capsys.readouterr().err


def test_benchmark_adds_relative_stats(csv_path, capsys):
    assert main(["run", csv_path, "--quantity", "10", "--benchmark", csv_path, "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert {"beta", "alpha", "information_ratio", "up_capture", "down_capture"} <= set(stats)
    assert stats["beta"] > 0  # a long position moves with its own benchmark


def test_benchmark_missing_dates_is_an_error(csv_path, tmp_path, capsys):
    short = pd.read_csv(csv_path).head(20)
    path = tmp_path / "short.csv"
    short.to_csv(path, index=False)
    assert main(["run", csv_path, "--benchmark", str(path)]) == 1
    assert "benchmark has no price" in capsys.readouterr().err


def test_no_benchmark_means_no_relative_stats(csv_path, capsys):
    assert main(["run", csv_path, "--json"]) == 0
    assert "beta" not in json.loads(capsys.readouterr().out)


def test_start_end_restrict_the_equity_curve(csv_path, tmp_path):
    out = tmp_path / "equity.csv"
    argv = ["run", csv_path, "--start", "2024-01-11", "--end", "2024-01-30", "--equity-csv", str(out)]
    assert main(argv) == 0
    dates = pd.read_csv(out)["date"]
    assert dates.iloc[0] == "2024-01-11" and dates.iloc[-1] == "2024-01-30"
    assert len(dates) == 20


def test_start_only_and_end_only(csv_path, tmp_path):
    out = tmp_path / "equity.csv"
    assert main(["run", csv_path, "--start", "2024-02-01", "--equity-csv", str(out)]) == 0
    assert pd.read_csv(out)["date"].iloc[0] == "2024-02-01"
    assert main(["run", csv_path, "--end", "2024-01-20", "--equity-csv", str(out)]) == 0
    assert pd.read_csv(out)["date"].iloc[-1] == "2024-01-20"


def test_empty_or_inverted_date_range_is_an_error(csv_path, capsys):
    assert main(["run", csv_path, "--start", "2030-01-01"]) == 1
    assert "no price data" in capsys.readouterr().err
    assert main(["run", csv_path, "--start", "2024-02-01", "--end", "2024-01-01"]) == 1
    assert "after --end" in capsys.readouterr().err


def test_run_json_bollinger(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "bollinger", "--bb-window", "5", "--bb-std", "1", "--json"]
    assert main(argv) == 0
    assert "sharpe" in capsys.readouterr().out


def test_bad_bollinger_params_return_error(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "bollinger", "--bb-std", "0"]
    assert main(argv) == 1
    assert "error" in capsys.readouterr().err


def test_run_json_macd(csv_path, capsys):
    argv = ["run", csv_path, "--strategy", "macd", "--macd-fast", "2", "--macd-slow", "4", "--macd-signal", "2", "--json"]
    assert main(argv) == 0
    assert "sharpe" in capsys.readouterr().out


def test_bad_macd_params_return_error(csv_path, capsys):
    assert main(["run", csv_path, "--strategy", "macd", "--macd-fast", "30"]) == 1
    assert "error" in capsys.readouterr().err


def test_risk_free_lowers_sharpe(csv_path, capsys):
    base = ["run", csv_path, "--strategy", "sma", "--fast", "3", "--slow", "8", "--json"]
    assert main(base) == 0
    zero = json.loads(capsys.readouterr().out)
    assert main(base + ["--risk-free", "0.05"]) == 0
    with_rf = json.loads(capsys.readouterr().out)
    assert with_rf["sharpe"] < zero["sharpe"]
    assert with_rf["sortino"] < zero["sortino"]
    assert with_rf["max_drawdown"] == zero["max_drawdown"]


def test_run_json_stochastic(csv_path, capsys):
    assert main(["run", csv_path, "--strategy", "stochastic", "--stoch-window", "5", "--json"]) == 0


def test_bad_stochastic_params_return_error(csv_path, capsys):
    assert main(["run", csv_path, "--strategy", "stochastic", "--stoch-oversold", "90"]) == 1
