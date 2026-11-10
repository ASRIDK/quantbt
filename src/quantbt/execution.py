"""Execution simulator that fills market orders with trading costs."""
from __future__ import annotations

from .events import FillEvent, MarketEvent, OrderEvent


class SimulatedExecution:
    """Fill market orders at a bar price, adjusted for costs.

    ``slippage_bps`` moves the fill price against the trader (buys pay more,
    sells receive less). Commission is ``commission_bps`` of traded notional
    plus a flat ``commission_per_order``.
    """

    def __init__(
        self,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        commission_per_order: float = 0.0,
    ) -> None:
        if min(commission_bps, slippage_bps, commission_per_order) < 0:
            raise ValueError("costs must be non-negative")
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.commission_per_order = commission_per_order

    def execute(self, order: OrderEvent, bar: MarketEvent, use_open: bool = True) -> FillEvent:
        """Fill ``order`` against ``bar`` (at its open by default, else close)."""
        if order.symbol != bar.symbol:
            raise ValueError("order and bar symbols differ")
        base = bar.open if use_open else bar.close
        sign = 1.0 if order.quantity > 0 else -1.0 if order.quantity < 0 else 0.0
        price = base * (1.0 + sign * self.slippage_bps / 1e4)
        notional = abs(order.quantity) * price
        commission = 0.0
        if order.quantity != 0:
            commission = notional * self.commission_bps / 1e4 + self.commission_per_order
        return FillEvent(order.timestamp, order.symbol, order.quantity, price, commission)
