"""Command-line interface: ``quantbt run prices.csv --strategy sma``."""
from __future__ import annotations

import argparse
import json
import math
import sys

from .backtest import Backtest
from .data import load_csv
from .execution import SimulatedExecution
from .portfolio import Portfolio
from .report import format_report, summarize
from .strategy import BuyAndHold, SmaCrossover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantbt", description="Event-driven backtesting")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="backtest a strategy on an OHLCV CSV file")
    run.add_argument("csv", help="path to OHLCV CSV file")
    run.add_argument("--strategy", choices=["buy-and-hold", "sma"], default="buy-and-hold")
    run.add_argument("--quantity", type=float, default=1.0, help="position size in units")
    run.add_argument("--fast", type=int, default=10, help="fast SMA window (sma only)")
    run.add_argument("--slow", type=int, default=30, help="slow SMA window (sma only)")
    run.add_argument("--cash", type=float, default=100_000.0, help="initial cash")
    run.add_argument("--commission-bps", type=float, default=0.0)
    run.add_argument("--slippage-bps", type=float, default=0.0)
    run.add_argument("--commission-per-order", type=float, default=0.0)
    run.add_argument("--periods-per-year", type=int, default=252)
    run.add_argument("--confidence", type=float, default=0.95, help="VaR/CVaR confidence")
    run.add_argument("--json", action="store_true", help="print statistics as JSON")
    return parser


def _run(args: argparse.Namespace) -> str:
    data = load_csv(args.csv)
    if args.strategy == "sma":
        strategy = SmaCrossover(args.fast, args.slow, args.quantity)
    else:
        strategy = BuyAndHold(args.quantity)
    execution = SimulatedExecution(args.commission_bps, args.slippage_bps, args.commission_per_order)
    bt = Backtest(data, strategy, portfolio=Portfolio(args.cash), execution=execution)
    pf = bt.run()
    stats = summarize(pf.returns(), args.periods_per_year, args.confidence)
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
