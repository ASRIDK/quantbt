import pytest

from quantbt.metrics import max_drawdown, sharpe_ratio


def test_sharpe_constant_returns_is_zero():
    assert sharpe_ratio([0.01] * 10) == 0.0


def test_max_drawdown_known_case():
    assert max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)


def test_empty_inputs():
    assert max_drawdown([]) == 0.0
