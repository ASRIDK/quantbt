"""Summarise a return series into a flat dict of headline statistics."""
from __future__ import annotations

from . import metrics


def summarize(returns, periods_per_year: int = 252, confidence: float = 0.95) -> dict[str, float]:
    """Return headline performance and risk statistics for ``returns``."""
    return {
        "annualized_return": metrics.annualized_return(returns, periods_per_year),
        "annualized_volatility": metrics.annualized_volatility(returns, periods_per_year),
        "sharpe": metrics.sharpe_ratio(returns, periods_per_year),
        "sortino": metrics.sortino_ratio(returns, periods_per_year),
        "max_drawdown": metrics.max_drawdown(returns),
        "max_drawdown_duration": float(metrics.max_drawdown_duration(returns)),
        "calmar": metrics.calmar_ratio(returns, periods_per_year),
        "var": metrics.value_at_risk(returns, confidence),
        "cvar": metrics.cvar(returns, confidence),
    }


def format_report(stats: dict[str, float]) -> str:
    """Render ``summarize`` output as an aligned plain-text table."""
    width = max(len(k) for k in stats)
    return "\n".join(f"{k:<{width}}  {v:>10.4f}" for k, v in stats.items())
