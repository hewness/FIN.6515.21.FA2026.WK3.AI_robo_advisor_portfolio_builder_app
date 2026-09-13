"""Plain-text notes that put a recommendation in the context of the client's goal and inputs."""

from __future__ import annotations

from .schemas import FinancialGoal, PortfolioRecommendation, PortfolioRequest, ProfileSummary

SHORT_GOAL_HORIZON_YEARS = 5
CONCENTRATION_THRESHOLD = 0.60
LOW_GOAL_PROBABILITY = 0.25
LONG_SAVINGS_GOAL_YEARS = 15
RETIREMENT_AGE_WARNING = 75


def input_warnings(
    request: PortfolioRequest,
    profile: ProfileSummary,
    rule_based: PortfolioRecommendation,
    mean_variance: PortfolioRecommendation,
) -> list[str]:
    """Non-blocking warnings about input combinations worth a second look."""
    warnings: list[str] = []
    target = request.resolved_target

    if target <= request.initial_investment:
        warnings.append(
            f"Your goal target (${target:,.0f}) is at or below your initial investment "
            f"(${request.initial_investment:,.0f}), so it is already met."
        )
    else:
        probs = [p for p in (rule_based.probability_of_meeting_target,
                             mean_variance.probability_of_meeting_target) if p is not None]
        if probs and max(probs) < LOW_GOAL_PROBABILITY:
            warnings.append(
                f"The ${target:,.0f} goal looks hard to reach: {max(probs):.0%} chance at best. "
                "Consider a larger monthly contribution, a longer horizon, or a lower target."
            )

    if request.goal in (FinancialGoal.HOME_PURCHASE, FinancialGoal.EDUCATION) \
            and request.horizon_years > LONG_SAVINGS_GOAL_YEARS:
        warnings.append(
            f"A {request.horizon_years}-year horizon is unusually long for {request.goal.label.lower()} savings; "
            "double-check the horizon."
        )

    if request.goal is FinancialGoal.RETIREMENT and not request.is_retired \
            and request.age + request.horizon_years > RETIREMENT_AGE_WARNING:
        warnings.append(
            f"This horizon runs to age {request.age + request.horizon_years}. If you plan to retire earlier, "
            "shorten the horizon to your retirement date."
        )

    if request.annual_income == 0 and not request.is_retired:
        warnings.append(
            f"Annual income is $0 but you are younger than your retirement age ({request.retirement_age}). "
            "Enter your earnings so the research-informed model can value your future income."
        )

    if profile.effective_risk_tolerance < profile.risk_tolerance and request.horizon_years < 3:
        warnings.append(
            f"With only {request.horizon_years} year{'s' if request.horizon_years != 1 else ''} to invest, "
            f"risk was capped at {profile.effective_risk_tolerance:g} (you chose {profile.risk_tolerance:g})."
        )
    return warnings


def build_notes(
    request: PortfolioRequest,
    profile: ProfileSummary,
    rule_based: PortfolioRecommendation,
    mean_variance: PortfolioRecommendation,
    research_informed: PortfolioRecommendation | None = None,
) -> list[str]:
    notes = [_goal_note(request, rule_based)]
    if research_informed is not None:
        notes.append(_research_note(research_informed))

    if request.hump_glide_path:
        base = rule_based.details.get("base_equity_pct")
        band = mean_variance.details.get("equity_band")
        if isinstance(base, float) and profile.equity_target is not None:
            band_text = f"; mean-variance keeps equity funds between {band[0]:.0%} and {band[1]:.0%}" if band else ""
            notes.append(
                f"Hump-shaped glide path at age {request.age}: {base:.0%} base equity, {profile.equity_target:.0%} "
                f"after your risk level{band_text}."
            )

    if profile.effective_risk_tolerance != profile.risk_tolerance:
        notes.append(
            f"Your {request.horizon_years}-year horizon changed the risk level used from "
            f"{profile.risk_tolerance:g} to {profile.effective_risk_tolerance:g} ({profile.risk_band})."
        )

    if request.goal is FinancialGoal.RETIREMENT and request.age + request.horizon_years < 60:
        notes.append(
            f"Retirement is still a long way off (age {request.age + request.horizon_years} at the end of this horizon); "
            "there is room to stay growth-oriented and adjust later."
        )

    top = max(mean_variance.holdings, key=lambda h: h.weight)
    if top.weight > CONCENTRATION_THRESHOLD:
        notes.append(
            f"The mean-variance portfolio puts {top.weight:.0%} in {top.ticker}. It reflects historical returns since "
            f"the start of the estimation window and can be concentrated; the rule-based portfolio is more diversified."
        )

    projection = rule_based.projection
    if request.monthly_contribution > 0 and projection.final_p50 > 0:
        share = projection.total_contributed / projection.final_p50
        notes.append(
            f"In the rule-based median projection, contributions (${projection.total_contributed:,.0f}) make up "
            f"{min(share, 1):.0%} of the ${projection.final_p50:,.0f} ending value; the rest is investment growth."
        )

    notes.append(
        "Projections are simulations based on historical estimates, not guarantees. Actual returns will differ."
    )
    return notes


def _research_note(rec: PortfolioRecommendation) -> str:
    d = rec.details
    a = d.get("assumptions") or {}
    capped = " (capped at 100%)" if d["unclipped_equity_pct"] > 1 else ""
    return (
        f"Research-informed: {d['merton_share']:.0%} Merton share (risk aversion {d['risk_aversion']:.1f}, "
        f"{a.get('log_equity_premium', 0):.0%} equity premium, {a.get('equity_volatility', 0):.1%} stock volatility, "
        f"{a.get('log_risk_free', 0):.0%} real rate) × (1 + ${d['human_capital']:,.0f} human capital ÷ "
        f"${d['financial_wealth']:,.0f} savings) = {d['equity_pct']:.0%} equity{capped}. Human capital is future "
        f"earnings plus ${d['retirement_income']:,.0f}/yr retirement income from age {d['retirement_age']}, "
        "discounted with Practical Finance's age-varying rates."
    )


def _goal_note(request: PortfolioRequest, rule_based: PortfolioRecommendation) -> str:
    horizon = request.horizon_years
    if request.goal is FinancialGoal.RETIREMENT:
        equity = rule_based.details.get("equity_pct")
        equity_text = f" ({equity:.0%} equity at age {request.age})" if isinstance(equity, float) else ""
        trend = ("equity rises toward its mid-life peak, then eases to about 60% by retirement"
                 if request.hump_glide_path else "equity falls as you get older")
        return (
            f"Retirement: the rule-based portfolio follows an age-based lifecycle glide path{equity_text}. "
            f"Revisit the allocation each year; {trend}."
        )
    if request.goal in (FinancialGoal.HOME_PURCHASE, FinancialGoal.EDUCATION):
        purpose = "down payment" if request.goal is FinancialGoal.HOME_PURCHASE else "tuition"
        label = request.goal.label
        if horizon <= SHORT_GOAL_HORIZON_YEARS:
            return (
                f"{label} in {horizon} year{'s' if horizon != 1 else ''}: money needed on a fixed date has little time "
                f"to recover from a downturn. Consider keeping the {purpose} in short-term TIPS (VTIP) or cash."
            )
        return (
            f"{label} in {horizon} years: plan to shift toward bonds, short-term TIPS and cash as the date approaches "
            f"so the {purpose} is not exposed to a late market drop."
        )
    return (
        "General wealth building: a diversified, long-term portfolio. Rebalance periodically and keep contributing "
        "through market ups and downs."
    )
