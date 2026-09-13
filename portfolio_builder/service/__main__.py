"""Command-line interface for the portfolio service.

    python -m portfolio_builder.service --age 45 --risk moderate --horizon 20 --initial 100000 --monthly 1000 --goal retirement
    python -m portfolio_builder.service --age 30 --risk 8 --horizon 3 --goal home_purchase --json
"""

from __future__ import annotations

import argparse
import json
import sys

from .portfolio_service import InputValidationError, PortfolioServiceError, get_portfolio_service
from .schemas import PortfolioRecommendation


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
    print(f"projection after {p.points[-1].year} years (contributed ${p.total_contributed:,.0f}): "
          f"p10 ${p.final_p10:,.0f} | median ${p.final_p50:,.0f} | p90 ${p.final_p90:,.0f} | expected ${p.final_expected:,.0f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m portfolio_builder.service", description="Recommend portfolios")
    parser.add_argument("--risk", default="moderate", help="1-10 or conservative / moderate / aggressive")
    parser.add_argument("--horizon", type=int, default=20, help="Investment horizon in years (1-30)")
    parser.add_argument("--initial", type=float, default=100_000, help="Initial investment ($1,000-$10,000,000)")
    parser.add_argument("--monthly", type=float, default=500, help="Monthly contribution ($0-$50,000)")
    parser.add_argument("--goal", default="general_wealth", help="retirement, home_purchase, education, general_wealth")
    parser.add_argument("--age", type=int, default=40, help="Age (18-80)")
    parser.add_argument("--json", action="store_true", help="Print the full response as JSON")
    args = parser.parse_args(argv)

    service = get_portfolio_service()
    try:
        response = service.build_portfolio({
            "risk_tolerance": args.risk, "horizon_years": args.horizon, "initial_investment": args.initial,
            "monthly_contribution": args.monthly, "goal": args.goal, "age": args.age,
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
    print(f"Estimates: {md.estimation_start} to {md.estimation_end} ({md.observations} {md.frequency} obs), "
          f"risk-free {md.risk_free_rate:.2%}, data as of {md.data_as_of}")
    _print_recommendation(response.rule_based)
    _print_recommendation(response.mean_variance)
    print("=" * 88)
    print(f"Efficient frontier: {len(response.efficient_frontier.points)} points; references: "
          + ", ".join(f"{r.name} ({r.expected_return:.2%} / {r.volatility:.2%})"
                      for r in response.efficient_frontier.reference_portfolios))
    print("\nNotes:")
    for note in response.notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
