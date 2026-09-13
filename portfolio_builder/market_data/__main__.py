"""Command-line interface for the market data cache.

    python -m portfolio_builder.market_data status
    python -m portfolio_builder.market_data download [--tickers VTI VXUS]
    python -m portfolio_builder.market_data refresh  [--tickers VTI VXUS]
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from ..universe import get_history_proxies, get_tickers
from .service import get_market_data_service


def default_tickers() -> list[str]:
    """Universe funds plus the older ETFs used as backtest history proxies (SPY is also the benchmark)."""
    return list(dict.fromkeys([*get_tickers(), *get_history_proxies().values()]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m portfolio_builder.market_data", description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", help="Cache directory (default: $MARKET_DATA_CACHE_DIR or data/market_data)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log cache hits and downloads")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Show what is cached")
    for name, help_text in (("download", "Download tickers missing from the cache"), ("refresh", "Force re-download")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--tickers", nargs="+",
                       help="Tickers to process (default: the universe plus backtest history proxies)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    svc = get_market_data_service(args.cache_dir)
    print(f"Cache: {svc.cache.root}")

    if args.command == "status":
        with pd.option_context("display.width", 200, "display.max_columns", None):
            status = svc.cache_status(default_tickers())
            proxy_for = {proxy: fund for fund, proxy in get_history_proxies().items()}
            status["asset_class"] = [
                label if isinstance(label, str) else f"(history proxy for {proxy_for.get(ticker, '?')})"
                for ticker, label in status["asset_class"].items()
            ]
            print(status.to_string())
        return 0

    action = svc.ensure_cached if args.command == "download" else svc.refresh
    results = action(args.tickers or default_tickers())
    for ticker, outcome in results.items():
        print(f"{ticker:6} {outcome}")
    return 1 if any(o.startswith("failed") for o in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
