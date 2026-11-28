import pandas as pd
import pytest

from quantbt.backtest import Backtest
from quantbt.execution import SimulatedExecution
from quantbt.strategy import (
    Bollinger,
    BuyAndHold,
    Donchian,
    EmaCrossover,
    Macd,
    MeanReversion,
    Rsi,
    SmaCrossover,
    Stochastic,
)


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


def test_mean_reversion_buys_dip_and_exits_on_recovery():
    closes = [100, 101, 99, 100, 101, 100, 99, 100, 101, 100] + [80, 81, 90, 100, 105, 106]
    df = frame(closes, closes)
    bt = Backtest(df, MeanReversion(window=10, entry=1.5, exit=0.0, quantity=1))
    bt.run()
    assert [f.quantity for f in bt.fills] == [1.0, -1.0]


def test_mean_reversion_ignores_flat_prices_and_validates():
    df = frame([50] * 30, [50] * 30)
    bt = Backtest(df, MeanReversion(window=5))
    bt.run()
    assert bt.fills == []
    with pytest.raises(ValueError):
        MeanReversion(window=1)
    with pytest.raises(ValueError):
        MeanReversion(entry=0)
    with pytest.raises(ValueError):
        MeanReversion(entry=1.0, exit=1.0)


def test_rsi_buys_oversold_and_exits_overbought():
    closes = list(range(30, 10, -1)) + list(range(10, 35))
    df = frame(closes, closes)
    bt = Backtest(df, Rsi(window=5, oversold=20, overbought=80, quantity=1))
    bt.run()
    assert [f.quantity for f in bt.fills] == [1.0, -1.0]


def test_rsi_flat_prices_never_trade():
    df = frame([10] * 30, [10] * 30)
    bt = Backtest(df, Rsi(window=5))
    bt.run()
    assert bt.fills == []


def test_rsi_rejects_bad_params():
    with pytest.raises(ValueError):
        Rsi(window=1)
    with pytest.raises(ValueError):
        Rsi(oversold=70, overbought=30)
    with pytest.raises(ValueError):
        Rsi(overbought=101)


def test_donchian_enters_on_breakout_and_exits_on_breakdown():
    closes = [10] * 6 + [12] * 3 + [8] * 3
    df = frame(closes, closes)
    bt = Backtest(df, Donchian(entry=5, exit=3, quantity=2))
    bt.run()
    assert [f.quantity for f in bt.fills] == [2.0, -2.0]


def test_donchian_flat_prices_never_trade():
    df = frame([10] * 30, [10] * 30)
    bt = Backtest(df, Donchian(entry=5, exit=3))
    bt.run()
    assert bt.fills == []


def test_donchian_rejects_bad_params():
    with pytest.raises(ValueError):
        Donchian(entry=0)
    with pytest.raises(ValueError):
        Donchian(entry=5, exit=6)


def test_ema_crossover_goes_long_in_uptrend_and_flat_in_downtrend():
    up = list(range(10, 50))
    df = frame(up, up)
    bt = Backtest(df, EmaCrossover(fast=3, slow=8, quantity=2))
    pf = bt.run()
    assert pf.position("X") == 2
    down = list(range(50, 10, -1))
    pf = Backtest(frame(down, down), EmaCrossover(fast=3, slow=8, quantity=2)).run()
    assert pf.position("X") == 0


def test_ema_crossover_waits_for_slow_window_and_validates():
    df = frame([10] * 5, [10] * 5)
    assert not Backtest(df, EmaCrossover(fast=2, slow=8)).run().position("X")
    with pytest.raises(ValueError):
        EmaCrossover(fast=8, slow=8)
    with pytest.raises(ValueError):
        EmaCrossover(fast=0, slow=5)


def test_bollinger_buys_below_lower_band_and_exits_at_mean():
    closes = [10, 11] * 6 + [4, 8, 11, 11, 11]
    bt = Backtest(frame(closes, closes), Bollinger(window=8, num_std=1.5, quantity=3))
    bt.run()
    assert [f.quantity for f in bt.fills] == [3.0, -3.0]


def test_bollinger_flat_prices_never_trade():
    bt = Backtest(frame([10] * 30, [10] * 30), Bollinger(window=5))
    bt.run()
    assert bt.fills == []


def test_bollinger_rejects_bad_params():
    with pytest.raises(ValueError):
        Bollinger(window=1)
    with pytest.raises(ValueError):
        Bollinger(num_std=0)


def test_macd_goes_long_in_uptrend_and_flat_after_reversal():
    closes = [float(i) for i in range(1, 41)] + [40.0 - 2 * i for i in range(1, 21)]
    bt = Backtest(frame(closes, closes), Macd(fast=3, slow=6, signal=3, quantity=2))
    bt.run()
    qty = [f.quantity for f in bt.fills]
    assert qty[0] == 2.0
    assert qty[-1] == -2.0


def test_macd_waits_for_warmup_and_validates():
    bt = Backtest(frame([1.0] * 5 + [2.0], [1.0] * 5 + [2.0]), Macd(fast=3, slow=6, signal=3))
    bt.run()
    assert bt.fills == []
    with pytest.raises(ValueError):
        Macd(fast=6, slow=6)
    with pytest.raises(ValueError):
        Macd(signal=0)


def test_stochastic_buys_oversold_and_exits_overbought():
    closes = list(range(30, 10, -1)) + list(range(10, 35))
    bt = Backtest(frame(closes, closes), Stochastic(window=5, oversold=10, overbought=90, quantity=1))
    bt.run()
    assert [f.quantity for f in bt.fills] == [1.0, -1.0]


def test_stochastic_flat_prices_never_trade_and_params_validate():
    bt = Backtest(frame([10] * 30, [10] * 30), Stochastic(window=5))
    bt.run()
    assert bt.fills == []
    with pytest.raises(ValueError):
        Stochastic(window=1)
    with pytest.raises(ValueError):
        Stochastic(oversold=80, overbought=20)
