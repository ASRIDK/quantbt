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


def skewness(returns) -> float:
    """Sample skewness (Fisher-Pearson, population moments); 0.0 if undefined."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 3:
        return 0.0
    sd = r.std()
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((r - r.mean()) / sd) ** 3))


def excess_kurtosis(returns) -> float:
    """Excess kurtosis (0 for a normal distribution); 0.0 if undefined."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 4:
        return 0.0
    sd = r.std()
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((r - r.mean()) / sd) ** 4) - 3.0)


def tail_ratio(returns, percentile: float = 95.0) -> float:
    """Size of the right tail relative to the left tail.

    ``|P_percentile| / |P_(100-percentile)|`` of the returns. Returns 0.0 for
    empty input or when the left tail is zero.
    """
    if not 50.0 < percentile < 100.0:
        raise ValueError("percentile must be in (50, 100)")
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    left = abs(np.percentile(r, 100.0 - percentile))
    if left < 1e-12:
        return 0.0
    return float(abs(np.percentile(r, percentile)) / left)


def win_rate(returns) -> float:
    """Fraction of periods with a strictly positive return (0.0 for empty input)."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Omega ratio: gains above ``threshold`` divided by shortfalls below it.

    Returns ``inf`` when there are gains but no shortfall and 0.0 when there
    are neither (or the input is empty).
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    gains = np.maximum(r - threshold, 0.0).sum()
    shortfall = np.maximum(threshold - r, 0.0).sum()
    if shortfall == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / shortfall)


def ulcer_index(returns) -> float:
    """Ulcer index: root-mean-square of percentage drawdowns (as a fraction).

    Penalises both the depth and the duration of drawdowns. Returns 0.0 for
    empty input.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return float(np.sqrt(np.mean(drawdown**2)))


def _paired(returns, benchmark) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    if r.shape != b.shape:
        raise ValueError("returns and benchmark must have the same length")
    return r, b


def beta(returns, benchmark) -> float:
    """Sensitivity of ``returns`` to ``benchmark`` (cov / benchmark variance).

    Returns 0.0 for fewer than two observations or a constant benchmark.
    """
    r, b = _paired(returns, benchmark)
    if len(r) < 2:
        return 0.0
    var = b.var(ddof=1)
    if var < 1e-18:
        return 0.0
    return float(np.cov(r, b, ddof=1)[0, 1] / var)


def alpha(returns, benchmark, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Annualised Jensen's alpha versus ``benchmark``."""
    r, b = _paired(returns, benchmark)
    if len(r) < 2:
        return 0.0
    rf = risk_free / periods_per_year
    bt = beta(r, b)
    return float(((r - rf).mean() - bt * (b - rf).mean()) * periods_per_year)


def information_ratio(returns, benchmark, periods_per_year: int = 252) -> float:
    """Annualised active return over tracking error (0.0 if tracking error is ~0)."""
    r, b = _paired(returns, benchmark)
    if len(r) < 2:
        return 0.0
    active = r - b
    sd = active.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def _capture(returns, benchmark, mask_fn) -> float:
    r, b = _paired(returns, benchmark)
    sel = mask_fn(b)
    if not sel.any():
        return 0.0
    # Geometric mean return in the selected periods, strategy over benchmark.
    rs = np.prod(1.0 + r[sel]) ** (1.0 / sel.sum()) - 1.0
    bs = np.prod(1.0 + b[sel]) ** (1.0 / sel.sum()) - 1.0
    if abs(bs) < 1e-12:
        return 0.0
    return float(rs / bs)


def up_capture(returns, benchmark) -> float:
    """Share of the benchmark's gains captured in periods when it rose.

    Ratio of the strategy's to the benchmark's geometric mean return over
    periods with a positive benchmark return. Above 1 means the strategy
    gained more than the benchmark in up markets. Returns 0.0 if the
    benchmark never rose.
    """
    return _capture(returns, benchmark, lambda b: b > 0)


def down_capture(returns, benchmark) -> float:
    """Share of the benchmark's losses suffered in periods when it fell.

    Like :func:`up_capture` over periods with a negative benchmark return.
    Below 1 means the strategy lost less than the benchmark in down
    markets. Returns 0.0 if the benchmark never fell.
    """
    return _capture(returns, benchmark, lambda b: b < 0)


def _check_window(returns, window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("window must be at least 2")
    return np.asarray(returns, dtype=float)


def rolling_volatility(returns, window: int, periods_per_year: int = 252) -> np.ndarray:
    """Annualised sample std over a trailing ``window``.

    The result has the same length as ``returns``; the first ``window - 1``
    entries are NaN because the window is not yet full.
    """
    r = _check_window(returns, window)
    out = np.full(len(r), np.nan)
    for i in range(window - 1, len(r)):
        out[i] = r[i - window + 1 : i + 1].std(ddof=1) * np.sqrt(periods_per_year)
    return out


def rolling_sharpe(
    returns, window: int, periods_per_year: int = 252, risk_free: float = 0.0
) -> np.ndarray:
    """Sharpe ratio over a trailing ``window`` (NaN until the window is full).

    Windows with ~zero volatility yield 0.0, matching :func:`sharpe_ratio`.
    """
    r = _check_window(returns, window)
    out = np.full(len(r), np.nan)
    for i in range(window - 1, len(r)):
        out[i] = sharpe_ratio(r[i - window + 1 : i + 1], periods_per_year, risk_free)
    return out
