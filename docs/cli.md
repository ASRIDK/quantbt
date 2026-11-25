# CLI

```
quantbt run prices.csv --strategy sma --fast 10 --slow 30 --commission-bps 5 --json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--strategy` | `buy-and-hold` | `buy-and-hold`, `sma`, `ema`, `zscore`, `rsi`, `donchian` or `bollinger` |
| `--quantity` | `1.0` | position size in units |
| `--fast`, `--slow` | `10`, `30` | SMA windows (sma only) |
| `--ema-fast`, `--ema-slow` | `12`, `26` | EMA spans (ema only) |
| `--window`, `--entry`, `--exit` | `20`, `1.0`, `0.0` | z-score window and thresholds (zscore only) |
| `--rsi-window`, `--oversold`, `--overbought` | `14`, `30`, `70` | RSI window and thresholds (rsi only) |
| `--entry-window`, `--exit-window` | `20`, `10` | breakout and exit lookbacks (donchian only) |
| `--bb-window`, `--bb-std` | `20`, `2.0` | Bollinger window and band width in standard deviations (bollinger only) |
| `--equity-csv` | none | write the equity curve (`date,equity`) to this CSV path |
| `--benchmark` | none | benchmark OHLCV CSV; adds `beta`, `alpha`, `information_ratio`, `up_capture`, `down_capture` |
| `--start`, `--end` | none | restrict the backtest to this inclusive date range (`YYYY-MM-DD`) |
| `--cash` | `100000` | initial cash |
| `--commission-bps`, `--slippage-bps` | `0` | proportional costs |
| `--commission-per-order` | `0` | fixed cost per order |
| `--periods-per-year` | `252` | annualisation factor |
| `--confidence` | `0.95` | VaR/CVaR confidence |
| `--json` | off | print statistics as JSON |
