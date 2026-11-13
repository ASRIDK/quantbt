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
    }  # fmt: skip
    text = format_report(stats)
    assert len(text.splitlines()) == len(stats)
    assert "sharpe" in text
