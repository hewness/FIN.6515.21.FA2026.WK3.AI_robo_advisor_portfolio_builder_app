"""Plain-text notes that put a recommendation in the context of the client's goal and inputs."""

from __future__ import annotations

from .schemas import FinancialGoal, PortfolioRecommendation, PortfolioRequest, ProfileSummary

SHORT_GOAL_HORIZON_YEARS = 5
CONCENTRATION_THRESHOLD = 0.60


def build_notes(
    request: PortfolioRequest,
    profile: ProfileSummary,
    rule_based: PortfolioRecommendation,
    mean_variance: PortfolioRecommendation,
) -> list[str]:
    notes = [_goal_note(request, rule_based)]

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


def _goal_note(request: PortfolioRequest, rule_based: PortfolioRecommendation) -> str:
    horizon = request.horizon_years
    if request.goal is FinancialGoal.RETIREMENT:
        equity = rule_based.details.get("equity_pct")
        equity_text = f" ({equity:.0%} equity at age {request.age})" if isinstance(equity, float) else ""
        return (
            f"Retirement: the rule-based portfolio follows an age-based lifecycle glide path{equity_text}. "
            "Revisit the allocation each year; equity falls as you get older."
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
