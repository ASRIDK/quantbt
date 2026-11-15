# CLI

```
quantbt run prices.csv --strategy sma --fast 10 --slow 30 --commission-bps 5 --json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--strategy` | `buy-and-hold` | `buy-and-hold`, `sma` or `zscore` |
| `--quantity` | `1.0` | position size in units |
| `--fast`, `--slow` | `10`, `30` | SMA windows (sma only) |
| `--window`, `--entry`, `--exit` | `20`, `1.0`, `0.0` | z-score window and thresholds (zscore only) |
| `--cash` | `100000` | initial cash |
| `--commission-bps`, `--slippage-bps` | `0` | proportional costs |
| `--commission-per-order` | `0` | fixed cost per order |
| `--periods-per-year` | `252` | annualisation factor |
| `--confidence` | `0.95` | VaR/CVaR confidence |
| `--json` | off | print statistics as JSON |
