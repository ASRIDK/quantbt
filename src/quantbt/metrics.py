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
