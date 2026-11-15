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


class MeanReversion(Strategy):
    """Long ``quantity`` when price is stretched below its rolling mean.

    Computes the z-score of the latest close against the last ``window``
    closes. Enters long when the z-score drops below ``-entry`` and exits
    (goes flat) once it recovers above ``-exit``. Between those thresholds
    the position is left unchanged.
    """

    def __init__(
        self, window: int = 20, entry: float = 1.0, exit: float = 0.0, quantity: float = 1.0
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if entry <= 0:
            raise ValueError("entry must be positive")
        if exit >= entry:
            raise ValueError("require exit < entry")
        self.window, self.entry, self.exit, self.quantity = window, entry, exit, quantity
        self._long = False

    def on_bar(self, bar: MarketEvent, history: pd.Series) -> SignalEvent | None:
        if len(history) < self.window:
            return None
        recent = history.iloc[-self.window :]
        sd = recent.std(ddof=1)
        if sd < 1e-12:
            return None
        z = (recent.iloc[-1] - recent.mean()) / sd
        if not self._long and z < -self.entry:
            self._long = True
            return SignalEvent(bar.timestamp, bar.symbol, self.quantity)
        if self._long and z > -self.exit:
            self._long = False
            return SignalEvent(bar.timestamp, bar.symbol, 0.0)
        return None
