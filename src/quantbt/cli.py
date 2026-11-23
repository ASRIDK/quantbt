"""Command-line interface: ``quantbt run prices.csv --strategy sma``."""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from . import __version__
from .backtest import Backtest
from .data import load_csv
from .execution import SimulatedExecution
from .portfolio import Portfolio
from .report import format_report, summarize
from .strategy import (
    BuyAndHold,
    Donchian,
    EmaCrossover,
    MeanReversion,
    Rsi,
    SmaCrossover,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantbt", description="Event-driven backtesting")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="backtest a strategy on an OHLCV CSV file")
    run.add_argument("csv", help="path to OHLCV CSV file")
    run.add_argument("--strategy", choices=["buy-and-hold", "sma", "zscore", "ema", "rsi", "donchian"], default="buy-and-hold")
    run.add_argument("--quantity", type=float, default=1.0, help="position size in units")
    run.add_argument("--fast", type=int, default=10, help="fast SMA window (sma only)")
    run.add_argument("--slow", type=int, default=30, help="slow SMA window (sma only)")
    run.add_argument("--ema-fast", type=int, default=12, help="fast EMA span (ema only)")
    run.add_argument("--ema-slow", type=int, default=26, help="slow EMA span (ema only)")
    run.add_argument("--window", type=int, default=20, help="lookback window (zscore only)")
    run.add_argument("--entry", type=float, default=1.0, help="entry z-score (zscore only)")
    run.add_argument("--exit", type=float, default=0.0, help="exit z-score (zscore only)")
    run.add_argument("--rsi-window", type=int, default=14, help="RSI lookback (rsi only)")
    run.add_argument("--oversold", type=float, default=30.0, help="RSI entry level (rsi only)")
    run.add_argument("--overbought", type=float, default=70.0, help="RSI exit level (rsi only)")
    run.add_argument("--entry-window", type=int, default=20, help="breakout lookback (donchian only)")
    run.add_argument("--exit-window", type=int, default=10, help="exit lookback (donchian only)")
    run.add_argument("--start", metavar="DATE", help="first bar to use, inclusive (YYYY-MM-DD)")
    run.add_argument("--end", metavar="DATE", help="last bar to use, inclusive (YYYY-MM-DD)")
    run.add_argument("--cash", type=float, default=100_000.0, help="initial cash")
    run.add_argument("--commission-bps", type=float, default=0.0)
    run.add_argument("--slippage-bps", type=float, default=0.0)
    run.add_argument("--commission-per-order", type=float, default=0.0)
    run.add_argument("--periods-per-year", type=int, default=252)
    run.add_argument("--confidence", type=float, default=0.95, help="VaR/CVaR confidence")
    run.add_argument("--equity-csv", metavar="PATH", help="write the equity curve to a CSV file")
    run.add_argument(
        "--benchmark",
        metavar="CSV",
        help="OHLCV CSV of a benchmark; adds beta, alpha and information ratio to the report",
    )
    run.add_argument("--json", action="store_true", help="print statistics as JSON")
    return parser


def _benchmark_returns(path: str, index) -> np.ndarray:
    """Close-to-close benchmark returns on the same dates as the equity curve."""
    close = load_csv(path)["close"]
    missing = index.difference(close.index)
    if len(missing):
        raise ValueError(f"benchmark has no price for {len(missing)} backtest date(s), first: {missing[0].date()}")
    return close.reindex(index).pct_change().dropna().to_numpy(dtype=float)


def _slice_dates(data, start: str | None, end: str | None):
    """Restrict ``data`` to ``[start, end]`` (inclusive); error if nothing is left."""
    lo = pd.Timestamp(start) if start else None
    hi = pd.Timestamp(end) if end else None
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"--start {start} is after --end {end}")
    out = data.loc[lo:hi]
    if out.empty:
        raise ValueError("no price data in the requested date range")
    return out


def _run(args: argparse.Namespace) -> str:
    data = _slice_dates(load_csv(args.csv), args.start, args.end)
    if args.strategy == "sma":
        strategy = SmaCrossover(args.fast, args.slow, args.quantity)
    elif args.strategy == "ema":
        strategy = EmaCrossover(args.ema_fast, args.ema_slow, args.quantity)
    elif args.strategy == "zscore":
        strategy = MeanReversion(args.window, args.entry, args.exit, args.quantity)
    elif args.strategy == "rsi":
        strategy = Rsi(args.rsi_window, args.oversold, args.overbought, args.quantity)
    elif args.strategy == "donchian":
        strategy = Donchian(args.entry_window, args.exit_window, args.quantity)
    else:
        strategy = BuyAndHold(args.quantity)
    execution = SimulatedExecution(args.commission_bps, args.slippage_bps, args.commission_per_order)
    bt = Backtest(data, strategy, portfolio=Portfolio(args.cash), execution=execution)
    pf = bt.run()
    if args.equity_csv:
        pf.equity_curve().to_csv(args.equity_csv, index_label="date")
    returns = pf.returns()
    benchmark = _benchmark_returns(args.benchmark, pf.equity_curve().index) if args.benchmark else None
    stats = summarize(returns, args.periods_per_year, args.confidence, benchmark)
    if args.json:
        # JSON has no infinity; report e.g. an infinite profit factor as null.
        safe = {k: (v if math.isfinite(v) else None) for k, v in stats.items()}
        return json.dumps(safe, indent=2)
    return format_report(stats)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(_run(args))
    except (OSError, ValueError) as exc:
        print(f"quantbt: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
