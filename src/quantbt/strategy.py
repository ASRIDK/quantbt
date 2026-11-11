"""Strategy API and example strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .events import MarketEvent, SignalEvent


class Strategy(ABC):
    """Base class for strategies.

    ``on_bar`` is called once per bar with the new :class:`MarketEvent` and
    the close-price history seen so far (including the current bar). It
    returns a :class:`SignalEvent` with the desired position, or ``None`` to
    leave the position unchanged.
    """

    @abstractmethod
    def on_bar(self, bar: MarketEvent, history: pd.Series) -> SignalEvent | None:
        ...


class BuyAndHold(Strategy):
    """Buy ``quantity`` on the first bar and hold."""

    def __init__(self, quantity: float = 1.0) -> None:
        self.quantity = quantity
        self._entered = False

    def on_bar(self, bar: MarketEvent, history: pd.Series) -> SignalEvent | None:
        if self._entered:
            return None
        self._entered = True
        return SignalEvent(bar.timestamp, bar.symbol, self.quantity)


class SmaCrossover(Strategy):
    """Long ``quantity`` when the fast SMA is above the slow SMA, else flat."""

    def __init__(self, fast: int = 10, slow: int = 30, quantity: float = 1.0) -> None:
        if not 0 < fast < slow:
            raise ValueError("require 0 < fast < slow")
        self.fast, self.slow, self.quantity = fast, slow, quantity

    def on_bar(self, bar: MarketEvent, history: pd.Series) -> SignalEvent | None:
        if len(history) < self.slow:
            return None
        fast = history.iloc[-self.fast :].mean()
        slow = history.iloc[-self.slow :].mean()
        target = self.quantity if fast > slow else 0.0
        return SignalEvent(bar.timestamp, bar.symbol, target)
