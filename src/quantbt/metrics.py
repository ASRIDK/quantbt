"""Performance and risk metrics on a series of periodic returns."""
from __future__ import annotations

import numpy as np


def sharpe_ratio(returns, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float) - risk_free / periods_per_year
    sd = r.std(ddof=1)
    if len(r) < 2 or sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min()) if len(equity) else 0.0


def sortino_ratio(
    returns,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
    target: float = 0.0,
) -> float:
    """Annualised Sortino ratio: excess return over downside deviation.

    Downside deviation is the root-mean-square of returns below ``target``
    (measured per period). Returns 0.0 when there is no downside or fewer
    than two observations.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd < 1e-12:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def value_at_risk(returns, confidence: float = 0.95) -> float:
    """Historical VaR, reported as a positive loss fraction.

    The loss that is exceeded in only ``1 - confidence`` of periods.
    Returns 0.0 for empty input or if the quantile is a gain.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    return float(max(-np.quantile(r, 1.0 - confidence), 0.0))


def cvar(returns, confidence: float = 0.95) -> float:
    """Historical CVaR (expected shortfall) as a positive loss fraction.

    Mean loss over the periods at or beyond the VaR threshold.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    cutoff = np.quantile(r, 1.0 - confidence)
    return float(max(-r[r <= cutoff].mean(), 0.0))


def annualized_return(returns, periods_per_year: int = 252) -> float:
    """Compound annual growth rate implied by periodic returns."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / len(r)) - 1.0


def annualized_volatility(returns, periods_per_year: int = 252) -> float:
    """Annualised sample standard deviation of returns."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def calmar_ratio(returns, periods_per_year: int = 252) -> float:
    """Annualised return divided by the absolute max drawdown (0.0 if no drawdown)."""
    dd = abs(max_drawdown(returns))
    if dd < 1e-12:
        return 0.0
    return annualized_return(returns, periods_per_year) / dd


def max_drawdown_duration(returns) -> int:
    """Longest stretch, in periods, spent below a previous equity peak."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])  # start from initial capital
    peak = np.maximum.accumulate(equity)
    longest = current = 0
    for under in equity[1:] < peak[1:]:
        current = current + 1 if under else 0
        longest = max(longest, current)
    return longest


def profit_factor(returns) -> float:
    """Gross gains divided by gross losses.

    Returns ``inf`` when there are gains but no losses and 0.0 when there
    are neither.
    """
    r = np.asarray(returns, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)
