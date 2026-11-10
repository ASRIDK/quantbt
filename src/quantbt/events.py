"""Event types passed through the backtest event loop."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketEvent:
    """A new bar is available for ``symbol``."""

    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class SignalEvent:
    """A strategy's desired position for ``symbol`` as a target quantity."""

    timestamp: pd.Timestamp
    symbol: str
    target_quantity: float


@dataclass(frozen=True)
class OrderEvent:
    """A market order. ``quantity`` is signed: positive buys, negative sells."""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float


@dataclass(frozen=True)
class FillEvent:
    """An executed order, including the price paid and fees charged."""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    commission: float
