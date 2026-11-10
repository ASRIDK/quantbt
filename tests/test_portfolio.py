import pandas as pd
import pytest

from quantbt.events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from quantbt.execution import SimulatedExecution
from quantbt.metrics import max_drawdown
from quantbt.portfolio import Portfolio

TS = pd.Timestamp("2024-01-02")


def bar(open_=100.0, close=101.0, symbol="X"):
    return MarketEvent(TS, symbol, open_, max(open_, close), min(open_, close), close, 1000.0)


def test_slippage_hurts_both_sides():
    ex = SimulatedExecution(slippage_bps=10)
    buy = ex.execute(OrderEvent(TS, "X", 10), bar())
    sell = ex.execute(OrderEvent(TS, "X", -10), bar())
    assert buy.price == pytest.approx(100.1)
    assert sell.price == pytest.approx(99.9)


def test_commission_bps_plus_flat():
    ex = SimulatedExecution(commission_bps=10, commission_per_order=1.0)
    fill = ex.execute(OrderEvent(TS, "X", -10), bar())
    assert fill.commission == pytest.approx(1000 * 0.001 + 1.0)


def test_fill_at_close_and_symbol_mismatch():
    ex = SimulatedExecution()
    assert ex.execute(OrderEvent(TS, "X", 1), bar(), use_open=False).price == 101.0
    with pytest.raises(ValueError):
        ex.execute(OrderEvent(TS, "Y", 1), bar())


def test_negative_costs_rejected():
    with pytest.raises(ValueError):
        SimulatedExecution(slippage_bps=-1)


def test_order_from_signal_uses_delta():
    p = Portfolio()
    assert p.order_from_signal(SignalEvent(TS, "X", 5)).quantity == 5
    p.apply_fill(FillEvent(TS, "X", 5, 100.0, 0.0))
    assert p.order_from_signal(SignalEvent(TS, "X", 5)) is None
    assert p.order_from_signal(SignalEvent(TS, "X", 2)).quantity == -3


def test_fill_updates_cash_and_positions():
    p = Portfolio(10_000)
    p.apply_fill(FillEvent(TS, "X", 10, 100.0, 2.0))
    assert p.cash == pytest.approx(10_000 - 1000 - 2)
    assert p.position("X") == 10
    p.apply_fill(FillEvent(TS, "X", -10, 110.0, 2.0))
    assert p.position("X") == 0 and "X" not in p.positions
    assert p.cash == pytest.approx(10_000 + 100 - 4)
    assert p.total_commission == 4.0


def test_equity_curve_and_returns_feed_metrics():
    p = Portfolio(1000)
    p.apply_fill(FillEvent(TS, "X", 10, 100.0, 0.0))
    for i, px in enumerate([100.0, 110.0, 99.0]):
        p.mark_to_market(TS + pd.Timedelta(days=i), {"X": px})
    assert list(p.equity_curve()) == [1000, 1100, 990]
    assert p.returns() == pytest.approx([0.1, -0.1])
    assert max_drawdown(p.returns()) == pytest.approx(-0.1)


def test_empty_portfolio_curve():
    p = Portfolio()
    assert p.equity_curve().empty and len(p.returns()) == 0
    with pytest.raises(ValueError):
        Portfolio(0)
