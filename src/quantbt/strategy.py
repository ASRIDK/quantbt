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


class EmaCrossover(Strategy):
    """Long ``quantity`` when the fast EMA is above the slow EMA, else flat.

    Exponential moving averages (``adjust=False``) react faster to recent
    prices than the simple averages used by :class:`SmaCrossover`. Signals
    start once ``slow`` closes are available.
    """

    def __init__(self, fast: int = 12, slow: int = 26, quantity: float = 1.0) -> None:
        if not 0 < fast < slow:
            raise ValueError("require 0 < fast < slow")
        self.fast, self.slow, self.quantity = fast, slow, quantity

    def on_bar(self, bar: MarketEvent, history: pd.Series) -> SignalEvent | None:
        if len(history) < self.slow:
            return None
        fast = history.ewm(span=self.fast, adjust=False).mean().iloc[-1]
        slow = history.ewm(span=self.slow, adjust=False).mean().iloc[-1]
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


class Rsi(Strategy):
    """Long ``quantity`` when the RSI is oversold, flat once it is overbought.

    Uses Wilder's RSI over the last ``window`` close-to-close changes. Enters
    long when RSI falls below ``oversold`` and exits once it rises above
    ``overbought``. Between the thresholds the position is left unchanged.
    """

    def __init__(
        self,
        window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        quantity: float = 1.0,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if not 0 <= oversold < overbought <= 100:
            raise ValueError("require 0 <= oversold < overbought <= 100")
        self.window, self.oversold, self.overbought = window, oversold, overbought
        self.quantity = quantity
        self._long = False

    def _rsi(self, history: pd.Series) -> float:
        deltas = history.iloc[-(self.window + 1) :].diff().dropna()
        gain = deltas.clip(lower=0).mean()
        loss = (-deltas.clip(upper=0)).mean()
        if loss == 0:
            return 50.0 if gain == 0 else 100.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    def on_bar(self, bar: MarketEvent, history: pd.Series) -> SignalEvent | None:
        if len(history) <= self.window:
            return None
        rsi = self._rsi(history)
        if not self._long and rsi < self.oversold:
            self._long = True
            return SignalEvent(bar.timestamp, bar.symbol, self.quantity)
        if self._long and rsi > self.overbought:
            self._long = False
            return SignalEvent(bar.timestamp, bar.symbol, 0.0)
        return None


class Donchian(Strategy):
    """Donchian channel breakout on closing prices.

    Enters long ``quantity`` when the close exceeds the highest close of the
    previous ``entry`` bars and exits (goes flat) when it drops below the
    lowest close of the previous ``exit`` bars. The current bar is excluded
    from the channel so a breakout can actually be detected.
    """

    def __init__(self, entry: int = 20, exit: int = 10, quantity: float = 1.0) -> None:
        if entry < 1 or exit < 1:
            raise ValueError("entry and exit windows must be at least 1")
        if exit > entry:
            raise ValueError("require exit <= entry")
        self.entry_window, self.exit_window, self.quantity = entry, exit, quantity
        self._long = False

    def on_bar(self, bar: MarketEvent, history: pd.Series) -> SignalEvent | None:
        if len(history) <= self.entry_window:
            return None
        close = history.iloc[-1]
        prior = history.iloc[:-1]
        if not self._long and close > prior.iloc[-self.entry_window :].max():
            self._long = True
            return SignalEvent(bar.timestamp, bar.symbol, self.quantity)
        if self._long and close < prior.iloc[-self.exit_window :].min():
            self._long = False
            return SignalEvent(bar.timestamp, bar.symbol, 0.0)
        return None
