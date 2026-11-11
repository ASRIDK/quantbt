import pandas as pd
import pytest

from quantbt.backtest import Backtest
from quantbt.execution import SimulatedExecution
from quantbt.strategy import BuyAndHold, SmaCrossover


def frame(opens, closes):
    idx = pd.date_range("2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {"open": opens, "high": closes, "low": opens, "close": closes, "volume": 1.0}, index=idx
    ).astype(float)


def test_buy_and_hold_fills_next_open():
    df = frame([10, 11, 12, 13], [10, 12, 14, 16])
    bt = Backtest(df, BuyAndHold(quantity=10))
    pf = bt.run()
    assert len(bt.fills) == 1
    assert bt.fills[0].price == 11  # next bar's open, not the signal bar's
    assert pf.position("X") == 10
    assert pf.equity_curve().iloc[-1] == pytest.approx(100_000 + 10 * (16 - 11))


def test_costs_reduce_equity():
    df = frame([10, 10, 10], [10, 10, 10])
    pf = Backtest(df, BuyAndHold(5), execution=SimulatedExecution(commission_per_order=2.0)).run()
    assert pf.equity_curve().iloc[-1] == pytest.approx(100_000 - 2.0)


def test_sma_crossover_trades_trend_and_exits():
    closes = [10] * 5 + list(range(11, 21)) + list(range(19, 5, -1))
    df = frame(closes, closes)
    bt = Backtest(df, SmaCrossover(fast=2, slow=5, quantity=1))
    bt.run()
    qtys = [f.quantity for f in bt.fills]
    assert qtys == [1.0, -1.0]


def test_sma_validation_and_empty_data():
    with pytest.raises(ValueError):
        SmaCrossover(fast=5, slow=5)
    with pytest.raises(ValueError):
        Backtest(frame([], []), BuyAndHold())
