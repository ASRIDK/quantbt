import pytest

from quantbt.metrics import (
    cvar,
    excess_kurtosis,
    max_drawdown,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
)


def test_skewness_sign_and_degenerate():
    assert skewness([1, 1, 1, 10]) > 0
    assert skewness([-10, -1, -1, -1]) < 0
    assert skewness([1, 2, 3, 4, 5]) == pytest.approx(0.0)
    assert skewness([0.01, 0.02]) == 0.0
    assert skewness([0.01] * 5) == 0.0


def test_excess_kurtosis_known_values():
    # two-point symmetric distribution has kurtosis 1 -> excess -2
    assert excess_kurtosis([-1, 1, -1, 1]) == pytest.approx(-2.0)
    assert excess_kurtosis([1, 2, 3]) == 0.0
    assert excess_kurtosis([0.5] * 6) == 0.0


def test_tail_ratio():
    r = list(range(-10, 11))  # symmetric
    assert tail_ratio(r) == pytest.approx(1.0)
    assert tail_ratio([]) == 0.0
    assert tail_ratio([0.0, 0.0, 0.0]) == 0.0
    assert tail_ratio(list(range(-10, 21))) > 1.0
    with pytest.raises(ValueError):
        tail_ratio(r, percentile=40)


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


def test_win_rate():
    from quantbt.metrics import win_rate

    assert win_rate([0.01, -0.02, 0.0, 0.03]) == pytest.approx(0.5)
    assert win_rate([]) == 0.0
    assert win_rate([0.0, 0.0]) == 0.0


def test_omega_ratio():
    from quantbt.metrics import omega_ratio

    assert omega_ratio([0.03, -0.01, 0.01, -0.01]) == pytest.approx(2.0)
    assert omega_ratio([0.02, -0.02], threshold=0.01) == pytest.approx(0.01 / 0.03)
    assert omega_ratio([0.01, 0.02]) == float("inf")
    assert omega_ratio([0.0, 0.0]) == 0.0
    assert omega_ratio([]) == 0.0


def test_ulcer_index():
    from quantbt.metrics import ulcer_index

    assert ulcer_index([]) == 0.0
    assert ulcer_index([0.01, 0.02, 0.03]) == 0.0
    # equity 1, 0.9, 0.9 -> drawdowns 0, -0.1, -0.1
    assert ulcer_index([-0.1, 0.0]) == pytest.approx(((0.01 + 0.01) / 3) ** 0.5)
    assert ulcer_index([-0.2, 0.0]) > ulcer_index([-0.1, 0.0])


def test_beta_alpha_information_ratio():
    from quantbt.metrics import alpha, beta, information_ratio

    b = [0.01, -0.02, 0.015, 0.0, 0.01]
    assert beta([2 * x for x in b], b) == pytest.approx(2.0)
    assert beta([0.01, 0.02], [0.01, 0.01]) == 0.0  # constant benchmark
    assert beta([0.01], [0.01]) == 0.0
    # r = 2b + 0.001 -> alpha = 0.001 per period
    r = [2 * x + 0.001 for x in b]
    assert alpha(r, b, periods_per_year=252) == pytest.approx(0.001 * 252)
    assert information_ratio(b, b) == 0.0
    assert information_ratio([x + 0.001 for x in b], b) == 0.0  # constant active return
    assert information_ratio([0.02, 0.0, 0.02, 0.0], [0.0] * 4) > 0


def test_benchmark_metrics_length_mismatch():
    from quantbt.metrics import alpha, beta, information_ratio

    for fn in (beta, alpha, information_ratio):
        with pytest.raises(ValueError):
            fn([0.01, 0.02], [0.01])
