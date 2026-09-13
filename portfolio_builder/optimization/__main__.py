"""Command-line interface for the optimization engine.

    python -m portfolio_builder.optimization --age 40 --risk 6
    python -m portfolio_builder.optimization --age 30 --risk 8 --method mean_variance --frontier
    python -m portfolio_builder.optimization --age 55 --risk 4 --method mean_variance --objective max_sharpe
    python -m portfolio_builder.optimization --age 68 --risk 6.5 --method research_informed --wealth 500000 --retirement-income 30000
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .engine import get_optimization_engine
from .inputs import DEFAULT_RISK_FREE_RATE
from .mean_variance import OBJECTIVES
from .models import AllocationResult, InvestorProfile


def _print_result(result: AllocationResult, show_frontier: bool) -> None:
    print("=" * 78)
    print(result.summary().splitlines()[0])
    print(result.summary().splitlines()[1])
    for key in ("equity_pct", "target_volatility", "merton_share"):
        if key in result.details:
            print(f"{key}: {result.details[key]:.2%}")
    if "human_capital" in result.details:
        d = result.details
        print(f"human capital: ${d['human_capital']:,.0f} | wealth ${d['financial_wealth']:,.0f} | "
              f"risk aversion {d['risk_aversion']:.2f}")
    if result.details.get("note"):
        print(f"note: {result.details['note']}")
    print(f"estimation window: {result.details.get('estimation_window')}")
    print("\nWeights by ticker:")
    print(result.to_frame().to_string(formatters={"weight": "{:.2%}".format}))
    print("\nWeights by asset class:")
    print(result.asset_class_weights.map("{:.2%}".format).to_string())
    for name, ref in result.reference_portfolios.items():
        m = ref.metrics
        top = ", ".join(f"{t} {w:.0%}" for t, w in ref.weights[ref.weights > 0.005].sort_values(ascending=False).items())
        print(f"\nReference {name}: return {m.expected_return:.2%}, vol {m.volatility:.2%}, "
              f"Sharpe {m.sharpe_ratio:.2f} [{top}]")
    if show_frontier and result.frontier is not None:
        points = result.frontier.points
        step = max(1, len(points) // 10)
        sample = pd.concat([points.iloc[::step], points.iloc[[-1]]]).drop_duplicates()
        print(f"\nEfficient frontier ({len(points)} points, sampled):")
        print(sample.to_string(formatters={
            "expected_return": "{:.2%}".format, "volatility": "{:.2%}".format, "sharpe_ratio": "{:.2f}".format,
        }))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m portfolio_builder.optimization", description="Build a portfolio")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--risk", type=float, required=True, help="Risk tolerance 1 (conservative) to 10 (aggressive)")
    parser.add_argument("--method", default="all", help="rule_based, mean_variance, research_informed, or all (default)")
    parser.add_argument("--objective", choices=OBJECTIVES, help="Mean-variance objective (default: risk_tolerance)")
    parser.add_argument("--target", type=float, help="Target for target_volatility / target_return, e.g. 0.10")
    parser.add_argument("--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE,
                        help="Return on cash and Sharpe risk-free rate (default 0)")
    parser.add_argument("--lookback-years", type=float)
    parser.add_argument("--income", type=float, default=0.0, help="Annual labor income (research_informed)")
    parser.add_argument("--retirement-income", type=float, help="Annual retirement income (default 40%% of income)")
    parser.add_argument("--retirement-age", type=int, default=67)
    parser.add_argument("--wealth", type=float, default=100_000.0, help="Financial wealth (default 100,000)")
    parser.add_argument("--frontier", action="store_true", help="Print a sample of the efficient frontier")
    args = parser.parse_args(argv)

    engine = get_optimization_engine(risk_free_rate=args.risk_free_rate, lookback_years=args.lookback_years)
    profile = InvestorProfile(age=args.age, risk_tolerance=args.risk, annual_income=args.income,
                              retirement_income=args.retirement_income, retirement_age=args.retirement_age,
                              financial_wealth=args.wealth)
    methods = engine.available_methods() if args.method == "all" else [args.method]
    for method in methods:
        options = {}
        if method == "mean_variance" and args.objective:
            options = {"objective": args.objective, "target": args.target}
        _print_result(engine.optimize(method, profile, **options), args.frontier)
    return 0


if __name__ == "__main__":
    sys.exit(main())
