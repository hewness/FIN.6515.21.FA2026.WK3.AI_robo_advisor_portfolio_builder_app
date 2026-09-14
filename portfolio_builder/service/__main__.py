"""Command-line interface for the portfolio service.

    python -m portfolio_builder.service --age 45 --risk moderate --horizon 20 --initial 100000 --monthly 1000 --goal retirement
    python -m portfolio_builder.service --age 30 --risk 8 --horizon 3 --goal home_purchase --json
    python -m portfolio_builder.service --age 68 --risk moderate --horizon 20 --initial 500000 --income 0 --retirement-income 30000
"""

from __future__ import annotations

import argparse
import json
import sys

from .portfolio_service import InputValidationError, PortfolioServiceError, get_portfolio_service
from .schemas import PortfolioRecommendation, PortfolioRequest


def _print_recommendation(rec: PortfolioRecommendation) -> None:
    print("=" * 88)
    print(rec.title)
    print(f"expected return {rec.expected_return:.2%} | volatility {rec.volatility:.2%} | Sharpe {rec.sharpe_ratio:.2f}")
    for h in rec.holdings:
        print(f"  {h.ticker:5} {h.weight:7.2%}  ${h.amount:>14,.2f}  {h.asset_class:40} {h.name[:40]}")
    fp = rec.frontier_position
    print(f"frontier: on_frontier={fp.on_frontier} gap={fp.return_gap:+.2%} position={fp.position:.2f}"
          + (f" | {fp.note}" if fp.note else ""))
    p = rec.projection
    print(f"{p.method} projection after {p.points[-1].year} years (contributed ${p.total_contributed:,.0f}): "
          f"p25 ${p.final_p25:,.0f} | median ${p.final_p50:,.0f} | p75 ${p.final_p75:,.0f} | expected ${p.final_expected:,.0f}")
    print(f"probability of reaching ${p.target_amount:,.0f}: {p.probability_of_meeting_target:.0%} | "
          f"max historical drawdown {rec.max_drawdown:.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m portfolio_builder.service", description="Recommend portfolios")
    d = {name: field.default for name, field in PortfolioRequest.model_fields.items()}
    parser.add_argument("--risk", default=d["risk_tolerance"], help="1-10 or conservative / moderate / aggressive")
    parser.add_argument("--horizon", type=int, default=d["horizon_years"], help="Investment horizon in years (1-30)")
    parser.add_argument("--initial", type=float, default=d["initial_investment"],
                        help="Initial investment ($1,000-$10,000,000)")
    parser.add_argument("--monthly", type=float, default=d["monthly_contribution"], help="Monthly contribution ($0-$50,000)")
    parser.add_argument("--goal", default=d["goal"].value, help="retirement, home_purchase, education, general_wealth")
    parser.add_argument("--age", type=int, default=d["age"], help="Age (18-80)")
    parser.add_argument("--target", type=float, help="Goal target in $ (default depends on goal)")
    parser.add_argument("--backtest-years", type=int, default=d["backtest_years"], help="Backtest lookback, 10-20 years")
    parser.add_argument("--rebalance", default=d["rebalance"], help="monthly, quarterly or annual")
    parser.add_argument("--glide-path", choices=("hump", "linear"), default="hump" if d["hump_glide_path"] else "linear",
                        help="Equity glide path: hump-shaped (default) or linear 110 - age")
    parser.add_argument("--income", type=float, default=d["annual_income"], help="Annual income (0 if retired)")
    parser.add_argument("--retirement-income", type=float,
                        help="Social Security + pensions per year (default 40%% of income)")
    parser.add_argument("--retirement-age", type=int, default=d["retirement_age"], help="Retirement age (50-75)")
    parser.add_argument("--projection", choices=("monte_carlo", "simple_percentiles"), default=d["projection_method"],
                        help="Wealth projection method (default monte_carlo)")
    parser.add_argument("--json", action="store_true", help="Print the full response as JSON")
    args = parser.parse_args(argv)

    service = get_portfolio_service()
    try:
        response = service.build_portfolio({
            "risk_tolerance": args.risk, "horizon_years": args.horizon, "initial_investment": args.initial,
            "monthly_contribution": args.monthly, "goal": args.goal, "age": args.age,
            "target_amount": args.target, "backtest_years": args.backtest_years, "rebalance": args.rebalance,
            "hump_glide_path": args.glide_path == "hump", "annual_income": args.income,
            "retirement_income": args.retirement_income, "retirement_age": args.retirement_age,
            "projection_method": args.projection,
        })
    except InputValidationError as exc:
        for field, message in exc.field_errors.items():
            print(f"{field}: {message}", file=sys.stderr)
        return 2
    except PortfolioServiceError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
        return 0

    pr = response.profile
    md = response.market_data
    print(f"Age {pr.age} | goal {pr.goal_label} | horizon {pr.horizon_years}y | risk {pr.risk_tolerance_input} "
          f"-> effective {pr.effective_risk_tolerance:g} ({pr.risk_band})")
    print(f"{pr.horizon_adjustment}")
    print(f"Glide path: {pr.glide_path} -> equity target {pr.equity_target:.1%}")
    print(f"Human capital: ${pr.human_capital:,.0f} -> research-informed equity {pr.research_equity_target:.1%}")
    print(f"Estimates: {md.estimation_start} to {md.estimation_end} ({md.observations} {md.frequency} obs), "
          f"Sharpe risk-free rate {md.risk_free_rate:.2%}, cash return {md.cash_return:.2%}, data as of {md.data_as_of}")
    _print_recommendation(response.rule_based)
    _print_recommendation(response.mean_variance)
    _print_recommendation(response.research_informed)
    print("=" * 88)
    print(f"Efficient frontier: {len(response.efficient_frontier.points)} points; references: "
          + ", ".join(f"{r.name} ({r.expected_return:.2%} / {r.volatility:.2%})"
                      for r in response.efficient_frontier.reference_portfolios))
    b = response.backtest
    print(f"Backtest {b.start} to {b.end} ({b.years_covered:.1f}y, {b.rebalance} rebalancing), "
          f"${b.initial_value:,.0f} initial:")
    for m in b.metrics.values():
        print(f"  {m.label:15} final ${m.final_value:>12,.0f} | CAGR {m.cagr:6.2%} | vol {m.volatility:6.2%} | "
              f"Sharpe {m.sharpe_ratio:5.2f} | max drawdown {m.max_drawdown:7.2%} ({m.max_drawdown_date})")
    for label, items in (("Warnings", response.warnings), ("Backtest notes", b.notes)):
        if items:
            print(f"\n{label}:")
            for item in items:
                print(f"  - {item}")
    insight = response.research_insight
    print(f"\nResearch vs. popular wisdom: {insight.headline}")
    for c in insight.comparisons:
        print(f"  {c.label:26} {c.equity:6.1%}")
    for point in insight.points:
        print(f"  - {point}")
    print("\nNotes:")
    for note in response.notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
