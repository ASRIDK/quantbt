# quantbt

A small event-driven backtesting library with risk metrics.

Per bar, the engine fills orders decided on the previous bar at this bar's
open (no lookahead), marks the portfolio at the close, then lets the strategy
see the bar and emit a signal for the next bar.

Building blocks:

| Module | Purpose |
| --- | --- |
| `quantbt.data` | `load_csv`, `load_yfinance` OHLCV loaders |
| `quantbt.events` | market, signal, order and fill events |
| `quantbt.strategy` | `Strategy` base class, `BuyAndHold`, `SmaCrossover` |
| `quantbt.execution` | `SimulatedExecution` with commission and slippage |
| `quantbt.portfolio` | `Portfolio` cash/position tracking and returns |
| `quantbt.metrics` / `quantbt.report` | risk metrics and `summarize` |

Install with `pip install -e .` (add `[yfinance]` for the Yahoo loader).
