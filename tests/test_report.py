import pytest

from quantbt.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
)
from quantbt.report import format_report, summarize


def test_annualized_return_full_year():
    assert annualized_return([0.1, 0.1], periods_per_year=2) == pytest.approx(0.21)


def test_annualized_return_edge_cases():
    assert annualized_return([]) == 0.0
    assert annualized_return([-1.0, 0.5]) == -1.0


def test_volatility_and_short_input():
    assert annualized_volatility([0.01]) == 0.0
    assert annualized_volatility([0.01, -0.01], periods_per_year=1) == pytest.approx(0.0141421356)


def test_calmar():
    r = [0.1, -0.2, 0.05]
    expected = annualized_return(r) / abs(max_drawdown(r))
    assert calmar_ratio(r) == pytest.approx(expected)
    assert calmar_ratio([0.01, 0.02]) == 0.0  # no drawdown


def test_summarize_keys_and_format():
    stats = summarize([0.01, -0.02, 0.015, 0.0, -0.005])
    assert set(stats) == {
        "annualized_return", "annualized_volatility", "sharpe", "sortino",
        "max_drawdown", "max_drawdown_duration", "calmar", "var", "cvar",
        "win_rate", "profit_factor", "skewness", "excess_kurtosis", "tail_ratio",
        "omega", "ulcer_index",
    }  # fmt: skip
    text = format_report(stats)
    assert len(text.splitlines()) == len(stats)
    assert "sharpe" in text


def test_summarize_with_benchmark_adds_relative_stats():
    r = [0.01, -0.02, 0.015, 0.0, 0.03]
    stats = summarize(r, benchmark=r)
    assert stats["beta"] == pytest.approx(1.0)
    assert stats["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert stats["information_ratio"] == 0.0
    assert stats["up_capture"] == pytest.approx(1.0)
    assert stats["down_capture"] == pytest.approx(1.0)
    assert "beta" not in summarize(r)


def test_summarize_risk_free_passed_to_ratios():
    import numpy as np

    from quantbt.report import summarize

    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 200)
    b = rng.normal(0.0005, 0.01, 200)
    plain = summarize(r, benchmark=b)
    rf = summarize(r, benchmark=b, risk_free=0.04)
    assert rf["sharpe"] < plain["sharpe"]
    assert rf["sortino"] < plain["sortino"]
    assert rf["alpha"] != plain["alpha"]
    assert rf["annualized_volatility"] == plain["annualized_volatility"]
