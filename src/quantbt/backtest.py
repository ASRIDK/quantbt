"""Event loop tying data, strategy, execution and portfolio together."""
from __future__ import annotations

import pandas as pd

from .events import MarketEvent
from .execution import SimulatedExecution
from .portfolio import Portfolio
from .strategy import Strategy


class Backtest:
    """Run a single-symbol backtest over an OHLCV DataFrame.

    Per bar: orders decided on the previous bar fill at this bar's open
    (no lookahead), the portfolio is marked at this bar's close, then the
    strategy sees the bar and may emit a signal for the next bar.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        symbol: str = "X",
        portfolio: Portfolio | None = None,
        execution: SimulatedExecution | None = None,
    ) -> None:
        if data.empty:
            raise ValueError("data is empty")
        self.data = data
        self.strategy = strategy
        self.symbol = symbol
        self.portfolio = portfolio or Portfolio()
        self.execution = execution or SimulatedExecution()
        self.fills = []

    def run(self) -> Portfolio:
        pending = None
        closes: list[float] = []
        index: list[pd.Timestamp] = []
        for ts, row in self.data.iterrows():
            bar = MarketEvent(
                ts, self.symbol, row["open"], row["high"], row["low"], row["close"], row["volume"]
            )
            if pending is not None:
                fill = self.execution.execute(pending, bar)
                self.portfolio.apply_fill(fill)
                self.fills.append(fill)
                pending = None
            self.portfolio.mark_to_market(ts, {self.symbol: bar.close})
            closes.append(bar.close)
            index.append(ts)
            signal = self.strategy.on_bar(bar, pd.Series(closes, index=index))
            if signal is not None:
                pending = self.portfolio.order_from_signal(signal)
        return self.portfolio
