"""Portfolio accounting: cash, positions, equity curve and order generation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import FillEvent, OrderEvent, SignalEvent


class Portfolio:
    def __init__(self, initial_cash: float = 100_000.0) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, float] = {}
        self.total_commission = 0.0
        self._equity: list[tuple[pd.Timestamp, float]] = []

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def order_from_signal(self, signal: SignalEvent) -> OrderEvent | None:
        """Turn a target-quantity signal into an order, or None if already there."""
        delta = signal.target_quantity - self.position(signal.symbol)
        if abs(delta) < 1e-12:
            return None
        return OrderEvent(signal.timestamp, signal.symbol, delta)

    def apply_fill(self, fill: FillEvent) -> None:
        self.cash -= fill.quantity * fill.price + fill.commission
        self.total_commission += fill.commission
        qty = self.position(fill.symbol) + fill.quantity
        if abs(qty) < 1e-12:
            self.positions.pop(fill.symbol, None)
        else:
            self.positions[fill.symbol] = qty

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(q * prices[s] for s, q in self.positions.items())

    def mark_to_market(self, timestamp: pd.Timestamp, prices: dict[str, float]) -> float:
        """Record and return total equity at the given prices."""
        equity = self.cash + self.market_value(prices)
        self._equity.append((timestamp, equity))
        return equity

    def equity_curve(self) -> pd.Series:
        if not self._equity:
            return pd.Series(dtype=float, name="equity")
        idx, vals = zip(*self._equity)
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")

    def returns(self) -> np.ndarray:
        """Periodic returns of the equity curve, for use with ``quantbt.metrics``."""
        eq = self.equity_curve().to_numpy(dtype=float)
        if len(eq) < 2:
            return np.array([])
        return eq[1:] / eq[:-1] - 1.0
