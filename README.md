# quantbt

A small event-driven backtesting library with risk metrics, built in the open one PR at a time.

## Usage
```
quantbt run prices.csv --strategy sma --fast 10 --slow 30 --commission-bps 5 --json
```
`prices.csv` needs `date,open,high,low,close,volume` columns. Without `--json` a plain-text table is printed.

## Roadmap
- [x] Metrics: Sharpe, max drawdown
- [x] Data loader: CSV
- [x] Data loader: yfinance (`quantbt.data.load_yfinance`, `pip install quantbt[yfinance]`)
- [x] Event loop: market, signal, order, fill events
- [x] Portfolio and execution simulator with costs
- [x] Strategy API and example strategies
- [x] More risk metrics (Sortino, VaR, CVaR)
- [x] CLI and report generation
- [x] Docs site (`mkdocs serve`, sources in `docs/`)
