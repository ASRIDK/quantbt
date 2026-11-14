# Quickstart

```python
from quantbt.backtest import Backtest
from quantbt.data import load_csv
from quantbt.execution import SimulatedExecution
from quantbt.portfolio import Portfolio
from quantbt.report import format_report, summarize
from quantbt.strategy import SmaCrossover

data = load_csv("prices.csv")  # date,open,high,low,close,volume
bt = Backtest(
    data,
    SmaCrossover(fast=10, slow=30, quantity=10),
    portfolio=Portfolio(100_000.0),
    execution=SimulatedExecution(commission_bps=5, slippage_bps=1),
)
portfolio = bt.run()
print(format_report(summarize(portfolio.returns())))
```

`Backtest.run()` returns the `Portfolio`; `bt.fills` holds every executed fill.
