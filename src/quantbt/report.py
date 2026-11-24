"""Summarise a return series into a flat dict of headline statistics."""
from __future__ import annotations

from . import metrics


def summarize(
    returns,
    periods_per_year: int = 252,
    confidence: float = 0.95,
    benchmark=None,
) -> dict[str, float]:
    """Return headline performance and risk statistics for ``returns``.

    If ``benchmark`` returns (same length as ``returns``) are given, beta,
    alpha, the information ratio and up/down capture versus the benchmark are appended.
    """
    stats = {
        "annualized_return": metrics.annualized_return(returns, periods_per_year),
        "annualized_volatility": metrics.annualized_volatility(returns, periods_per_year),
        "sharpe": metrics.sharpe_ratio(returns, periods_per_year),
        "sortino": metrics.sortino_ratio(returns, periods_per_year),
        "max_drawdown": metrics.max_drawdown(returns),
        "max_drawdown_duration": float(metrics.max_drawdown_duration(returns)),
        "calmar": metrics.calmar_ratio(returns, periods_per_year),
        "var": metrics.value_at_risk(returns, confidence),
        "cvar": metrics.cvar(returns, confidence),
        "win_rate": metrics.win_rate(returns),
        "profit_factor": metrics.profit_factor(returns),
        "skewness": metrics.skewness(returns),
        "excess_kurtosis": metrics.excess_kurtosis(returns),
        "tail_ratio": metrics.tail_ratio(returns),
        "omega": metrics.omega_ratio(returns),
        "ulcer_index": metrics.ulcer_index(returns),
    }
    if benchmark is not None:
        stats["beta"] = metrics.beta(returns, benchmark)
        stats["alpha"] = metrics.alpha(returns, benchmark, periods_per_year)
        stats["information_ratio"] = metrics.information_ratio(returns, benchmark, periods_per_year)
        stats["up_capture"] = metrics.up_capture(returns, benchmark)
        stats["down_capture"] = metrics.down_capture(returns, benchmark)
    return stats


def format_report(stats: dict[str, float]) -> str:
    """Render ``summarize`` output as an aligned plain-text table."""
    width = max(len(k) for k in stats)
    return "\n".join(f"{k:<{width}}  {v:>10.4f}" for k, v in stats.items())
