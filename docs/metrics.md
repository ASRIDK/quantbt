# Metrics

All functions in `quantbt.metrics` take a sequence of periodic returns.

| Function | Description |
| --- | --- |
| `sharpe_ratio` | annualised excess return over volatility |
| `sortino_ratio` | excess return over downside deviation |
| `max_drawdown` | worst peak-to-trough loss (negative fraction) |
| `max_drawdown_duration` | longest time spent below a prior peak, in periods |
| `value_at_risk`, `cvar` | historical VaR / expected shortfall, as positive loss fractions |
| `annualized_return`, `annualized_volatility` | CAGR and annualised sample std |
| `calmar_ratio` | annualised return over absolute max drawdown |
| `profit_factor` | gross gains over gross losses |

`quantbt.report.summarize(returns)` bundles the headline statistics into a
dict and `format_report` renders it as a text table.
