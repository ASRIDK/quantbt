# Strategies

Subclass `quantbt.strategy.Strategy` and implement `on_bar`:

```python
import pandas as pd
from quantbt.events import MarketEvent, SignalEvent
from quantbt.strategy import Strategy

class Momentum(Strategy):
    def on_bar(self, bar: MarketEvent, history: pd.Series) -> SignalEvent | None:
        if len(history) < 2:
            return None
        target = 1.0 if history.iloc[-1] > history.iloc[-2] else 0.0
        return SignalEvent(bar.timestamp, bar.symbol, target)
```

`history` is the close-price series seen so far, including the current bar.
Return a `SignalEvent` carrying the desired position, or `None` to leave the
position unchanged. Built in: `BuyAndHold(quantity)` and
`SmaCrossover(fast, slow, quantity)`.
