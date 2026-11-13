import pytest

from quantbt.metrics import (
    cvar,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)


def test_sharpe_constant_returns_is_zero():
    assert sharpe_ratio([0.01] * 10) == 0.0


def test_max_drawdown_known_case():
    assert max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)


def test_empty_inputs():
    assert max_drawdown([]) == 0.0


def test_sortino_known_case():
    r = [0.02, -0.01, 0.03, -0.02]
    dd = (((0.01**2) + (0.02**2)) / 4) ** 0.5
    expected = (sum(r) / 4) / dd * (252**0.5)
    assert sortino_ratio(r) == pytest.approx(expected)


def test_sortino_no_downside_or_short_input_is_zero():
    assert sortino_ratio([0.01, 0.02, 0.03]) == 0.0
    assert sortino_ratio([-0.01]) == 0.0


def test_var_and_cvar_known_case():
    r = list(range(-10, 10))  # -10 .. 9, as percent-like units
    r = [x / 100 for x in r]
    assert value_at_risk(r, 0.95) == pytest.approx(0.0905)
    assert cvar(r, 0.95) == pytest.approx(0.10)
    assert cvar(r, 0.95) >= value_at_risk(r, 0.95)


def test_var_cvar_gains_only_and_empty():
    assert value_at_risk([0.01, 0.02, 0.03]) == 0.0
    assert cvar([0.01, 0.02, 0.03]) == 0.0
    assert value_at_risk([]) == 0.0
    assert cvar([]) == 0.0


def test_var_invalid_confidence():
    with pytest.raises(ValueError):
        value_at_risk([0.1], 1.0)
    with pytest.raises(ValueError):
        cvar([0.1], 0.0)


def test_max_drawdown_monotonic_gains_is_zero():
    assert max_drawdown([0.01, 0.02, 0.03]) == 0.0


def test_sharpe_risk_free_lowers_ratio():
    returns = [0.01, -0.005, 0.02, 0.0, 0.015]
    assert sharpe_ratio(returns, risk_free=0.05) < sharpe_ratio(returns)


def test_sharpe_single_observation_is_zero():
    assert sharpe_ratio([0.05]) == 0.0


def test_max_drawdown_duration():
    from quantbt.metrics import max_drawdown_duration

    assert max_drawdown_duration([]) == 0
    assert max_drawdown_duration([0.01, 0.02]) == 0
    # peak, then 3 periods below peak, recovery, then 1 period below
    assert max_drawdown_duration([0.1, -0.05, 0.01, 0.01, 0.1, -0.01]) == 3
    assert max_drawdown_duration([-0.01, -0.01, -0.01]) == 3


def test_profit_factor():
    from quantbt.metrics import profit_factor

    assert profit_factor([0.02, -0.01, 0.01, -0.01]) == pytest.approx(1.5)
    assert profit_factor([0.01, 0.02]) == float("inf")
    assert profit_factor([]) == 0.0
    assert profit_factor([-0.01]) == 0.0
