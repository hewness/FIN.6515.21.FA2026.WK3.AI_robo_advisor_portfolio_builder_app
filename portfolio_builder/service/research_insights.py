"""Where the research-informed portfolio disagrees with popular financial advice, and why.

Sources:
* Choi, J. J. (2022). "Popular Personal Financial Advice versus the Professors." Journal of Economic Perspectives.
* Choi, J. J., Liu, C. and Liu, P. (2025). "Practical Finance: An Approximate Solution to Lifecycle Portfolio Choice."
* Duarte, V., Fonseca, J., Goodman, A. S. and Parker, J. A. (2022). "Simple Allocation Rules and Optimal
  Portfolio Choice Over the Lifecycle." NBER Working Paper 29559.
"""

from __future__ import annotations

import numpy as np

from ..universe import EQUITY_ASSET_CLASSES, get_asset_class
from .schemas import EquityComparison, PortfolioRecommendation, PortfolioRequest, ProfileSummary, ResearchInsight

SOURCES = [
    "Choi, Liu & Liu (2025), Practical Finance: equity = Merton share × (1 + human capital ÷ savings)",
    "Choi (2022), Popular Personal Financial Advice versus the Professors, Journal of Economic Perspectives",
    "Duarte, Fonseca, Goodman & Parker (2022), Simple Allocation Rules and Optimal Portfolio Choice "
    "Over the Lifecycle, NBER WP 29559",
]
# Typical target-date fund glide path used by Duarte et al. (2022), modeled on Vanguard Target Retirement:
# 90% equity to age 40, down 1.5 points a year to 60% at 60, to 49.5% at 65, 32% at 70 and 30% from 71.
TDF_GLIDE_PATH: tuple[tuple[float, float], ...] = ((40, 0.90), (60, 0.60), (65, 0.495), (70, 0.32), (71, 0.30))
NEAR_RETIREMENT_YEARS = 5
HIGH_EQUITY = 0.90
LOW_HUMAN_CAPITAL_RATIO = 0.5
MAX_POINTS = 5


def typical_tdf_equity(age: float) -> float:
    """Equity share of a typical target-date fund at ``age`` (flat before 40 and after 71)."""
    ages, equity = zip(*TDF_GLIDE_PATH)
    return float(np.interp(age, ages, equity))


def age_rule_equity(age: float) -> float:
    """The popular "100 minus your age" rule of thumb."""
    return min(max((100 - age) / 100, 0.0), 1.0)


def equity_share(rec: PortfolioRecommendation) -> float:
    """Share of a recommendation held in equity asset classes (stocks and REITs)."""
    names = {get_asset_class(key).name for key in EQUITY_ASSET_CLASSES}
    return sum(a.weight for a in rec.asset_classes if a.asset_class in names)


def money(value: float) -> str:
    """Compact dollars: $950, $584k, $1.2M."""
    if abs(value) >= 999_500:
        return f"${value / 1e6:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1e3:.0f}k"
    return f"${value:,.0f}"


def build_research_insight(
    request: PortfolioRequest,
    profile: ProfileSummary,
    recommendations: dict[str, PortfolioRecommendation],
) -> ResearchInsight:
    research = recommendations["research_informed"]
    d = research.details
    equity = float(d["equity_pct"])
    merton = float(d["merton_share"])
    human_capital = float(d["human_capital"])
    wealth = float(d["financial_wealth"])
    ratio = human_capital / wealth
    age = request.age
    age_rule, tdf = age_rule_equity(age), typical_tdf_equity(age)

    comparisons = [
        EquityComparison(key="research_informed", label="Research-informed", equity=equity),
        EquityComparison(key="rule_based", label="Rule-based", equity=equity_share(recommendations["rule_based"])),
        EquityComparison(key="mean_variance", label="Mean-variance",
                         equity=equity_share(recommendations["mean_variance"])),
        EquityComparison(key="age_rule", label="100 − age rule", equity=age_rule),
        EquityComparison(key="target_date_fund", label="Typical target-date fund", equity=tdf),
    ]

    popular = (age_rule + tdf) / 2
    gap = equity - popular
    if abs(gap) < 0.05:
        direction = "close to"
    else:
        direction = "more stock than" if gap > 0 else "less stock than"
    headline = (f"Research-informed holds {equity:.0%} in stocks, {direction} popular age rules "
                f"(100 − age: {age_rule:.0%}; typical target-date fund: {tdf:.0%}).")

    points = [_human_capital_point(request, human_capital, wealth, ratio, equity)]
    if request.is_retired or request.retirement_age - age <= NEAR_RETIREMENT_YEARS:
        points.append(
            "Retirees still have long horizons and pension-like income. Duarte et al. (2022) find optimal equity "
            "stays near 60% at and through retirement, while target-date funds glide down to 30–40%; that simple "
            "age rule costs households roughly 2–3% of consumption a year."
        )
    if equity >= HIGH_EQUITY and not request.is_retired:
        points.append(
            f"Your human capital is {ratio:.1f}× your savings, so the formula holds up to 100% stocks. Practical "
            "Finance keeps a 45-year-old fully in stocks until savings pass about 1.6× income. Popular books disagree: "
            "14 of them warn against ever holding 100% stocks, and 29 keep money for near-term needs in cash (Choi 2022)."
        )
    if ratio < LOW_HUMAN_CAPITAL_RATIO:
        points.append(
            f"Your savings are large relative to your future income, so the portfolio carries most of your risk "
            f"by itself and equity falls toward the {merton:.0%} Merton share. Age-only rules cannot see this; "
            "Duarte et al. (2022) find wealth is one of the biggest drivers of optimal equity at a given age."
        )
    points.append(
        "31 of 45 popular books say stocks get safer the longer you hold them. The research model does not rely "
        "on time diversification: equity falls with age only because remaining human capital shrinks relative "
        "to savings (Choi 2022)."
    )
    points.append(
        "In Practical Finance's simulations, following 100 − age loses 2.0% of lifetime consumption and a constant "
        "60% loses 3.75%, versus 0.06% for this formula."
    )
    classes = d.get("asset_class_weights") or {}
    stocks = sum(classes.get(k, 0.0) for k in EQUITY_ASSET_CLASSES)
    if stocks > 0:
        abroad = (classes.get("intl_developed", 0.0) + classes.get("emerging_markets", 0.0)) / stocks
        points.append(
            f"Popular authors put about 27% of stocks abroad; the academic benchmark is market-cap weight "
            f"(about 59% non-US). The research portfolio holds {abroad:.0%} of its stocks outside the US (Choi 2022)."
        )

    return ResearchInsight(
        equity=equity,
        popular_equity=age_rule,
        comparisons=comparisons,
        headline=headline,
        points=points[:MAX_POINTS],
        sources=SOURCES,
    )


def _human_capital_point(request: PortfolioRequest, human_capital: float, wealth: float, ratio: float,
                         equity: float) -> str:
    benefit = request.resolved_retirement_income
    if request.is_retired:
        source = f"your {money(benefit)}/yr of Social Security and pensions"
    else:
        source = (f"your {money(request.annual_income)}/yr of earnings until {request.retirement_age} plus "
                  f"{money(benefit)}/yr in retirement")
    return (
        "Popular rules look only at age, and only 8 of 47 best-selling finance books mention human capital "
        f"(Choi 2022). The research model counts {source} as a bond-like asset worth about {money(human_capital)} "
        f"({ratio:.1f}× your {money(wealth)} of savings), so your savings can hold {equity:.0%} in stocks "
        "(Practical Finance)."
    )
