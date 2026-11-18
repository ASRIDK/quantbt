# Metrics

All functions in `quantbt.metrics` take a sequence of periodic returns.
The benchmark-relative metrics (`beta`, `alpha`, `information_ratio`) also take
a benchmark series of the same length.

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
| `skewness`, `excess_kurtosis` | shape of the return distribution (0 for a normal) |
| `tail_ratio` | 95th percentile gain over 5th percentile loss |
| `omega_ratio` | gains over shortfalls relative to a threshold (default 0) |
| `ulcer_index` | RMS of drawdowns; penalises depth and duration |
| `beta`, `alpha` | sensitivity to, and annualised Jensen's alpha versus, a benchmark return series |
| `information_ratio` | annualised active return over tracking error versus a benchmark |

`quantbt.report.summarize(returns)` bundles the headline statistics into a
dict and `format_report` renders it as a text table.
