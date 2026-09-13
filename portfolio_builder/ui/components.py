"""HTML snippets, tables and markdown for the dashboard."""

from __future__ import annotations

from html import escape

import pandas as pd

from ..service import PortfolioResponse
from ..service.research_insights import equity_share, money
from ..service.schemas import PortfolioRecommendation
from .charts import _money_short
from .theme import (
    ASSET_CLASS_COLORS,
    ASSET_CLASS_SHORT,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_TITLES,
    METHODS,
    POPULAR_RULE_COLOR,
)
PROJECTION_ICONS = {"Monte Carlo": "🎲", "Simple percentiles": "📐"}
HOLDINGS_COLUMNS = [" ", "Ticker", "Asset class", "Exp. ret.", "Weight", "Amount"]
HOLDINGS_DATATYPES = ["html", "str", "str", "str", "str", "str"]


def _tile(label: str, value: str, sub: str = "", tooltip: str = "", css: str = "") -> str:
    return (
        f"<div class='tile {css}' title='{escape(tooltip)}'>"
        f"<div class='label'>{escape(label)}</div><div class='value'>{escape(value)}</div>"
        f"<div class='sub'>{escape(sub)}</div></div>"
    )


def _probability_class(p: float | None) -> str:
    if p is None:
        return ""
    return "good" if p >= 0.70 else "fair" if p >= 0.40 else "poor"


def _card(resp: PortfolioResponse, method: str) -> str:
    rec: PortfolioRecommendation = resp.recommendation(method)
    proj = rec.projection
    bench = resp.backtest.metrics.get("benchmark")
    prob = rec.probability_of_meeting_target
    end_age = resp.request.age + resp.request.horizon_years
    tiles = [
        _tile("Exp. return", f"{rec.expected_return:.1%}", "per year",
              "Annualized expected return estimated from monthly returns over the estimation window."),
        _tile("Volatility", f"{rec.volatility:.1%}", "per year",
              "Annualized standard deviation of returns: how much the portfolio typically swings."),
        _tile("Sharpe", f"{rec.sharpe_ratio:.2f}", f"rf {resp.market_data.risk_free_rate:.1%}",
              "Return in excess of the risk-free rate per unit of volatility. Higher is better."),
        _tile("Max drawdown", f"{rec.max_drawdown:.1%}" if rec.max_drawdown is not None else "—",
              f"SPY {bench.max_drawdown:.1%}" if bench else "backtest",
              f"Largest peak-to-trough loss in the {resp.backtest.years_covered:.0f}-year backtest "
              f"({resp.backtest.start} to {resp.backtest.end}).", "neg"),
        _tile("Goal odds", f"{prob:.0%}" if prob is not None else "—",
              f"{_money_short(proj.target_amount)} by {end_age}" if proj.target_amount else "",
              f"Probability of reaching ${proj.target_amount or 0:,.0f} by age {end_age}: "
              + (f"share of {proj.simulations:,} simulated outcomes that end at or above the goal target."
                 if proj.method == "monte_carlo" else
                 "chance the annualized return over your horizon beats the constant return needed to reach the "
                 "goal (simple percentiles, no simulation)."),
              _probability_class(prob)),
    ]
    color = METHOD_COLORS[method]
    basis, basis_tip = equity_basis(resp, method)
    description, basis_css = rec.description, "basis"
    cap_note = equity_cap_note(resp, method)
    if cap_note:  # replaces the model description (still in the tooltip), so the card keeps its shape
        description, basis_css = cap_note, "basis capped"
    return (
        f"<div class='card-head'><span class='name' style='color:{color}'>{escape(rec.title)}</span>"
        f"<span class='desc' title='{escape(description + ' ' + rec.description if cap_note else description)}'>"
        f"{escape(description)}</span>"
        f"<span class='{basis_css}' title='{escape(basis_tip)}'>{escape(basis)}</span></div>"
        f"<div class='tiles'>{''.join(tiles)}</div>"
    )


def equity_basis(resp: PortfolioResponse, method: str) -> tuple[str, str]:
    """A short line on how the card's equity share was set, and a longer hover explanation.

    Every card shows the line in the same position, which keeps the three cards aligned.
    """
    rec = resp.recommendation(method)
    d = rec.details
    if method == "research_informed":
        h, w = money(d["human_capital"]), money(d["financial_wealth"])
        formula = f"Merton share {d['merton_share']:.0%} × (1 + {h} human capital ÷ {w} savings)"
        if equity_cap_note(resp, method):
            return (f"⚠ Capped at 100% stocks · formula gives {d['unclipped_equity_pct']:.0%}",
                    f"{formula} = {d['unclipped_equity_pct']:.0%}, capped at 100% (no borrowing to buy stocks). "
                    "The mix responds to risk, age and horizon again once savings grow relative to future income.")
        return (f"Equity {d['merton_share']:.0%} × (1 + {h} ÷ {w}) → {d['equity_pct']:.0%}",
                f"{formula} = {d['equity_pct']:.0%} equity")
    if method == "rule_based":
        glide = resp.profile.glide_path
        short = "hump glide path" if glide.lower().startswith("hump") else "110 − age rule"
        return (f"Equity {d['equity_pct']:.0%} · {short} at age {resp.request.age}",
                f"{glide} glide path at age {resp.request.age}, shifted by risk tolerance")
    band = d.get("equity_band")
    if band:
        equity = d.get("equity_weight", equity_share(rec))
        return (f"Equity {equity:.0%} · kept within {band[0]:.0%}–{band[1]:.0%}",
                f"Equity funds held within ±5 points of the glide path's {d.get('equity_target', equity):.0%}")
    equity = equity_share(rec)
    return f"Equity {equity:.0%} · set by the frontier", "Equity share wherever the efficient frontier puts it (no age input)"


def equity_cap_note(resp: PortfolioResponse, method: str) -> str | None:
    """When the research formula asks for more than 100% stocks, explain why the mix stops responding."""
    if method != "research_informed":
        return None
    d = resp.recommendation(method).details
    if d.get("unclipped_equity_pct", 0) <= 1:
        return None
    return (f"Human capital ({money(d['human_capital'])}) dwarfs savings ({money(d['financial_wealth'])}): risk, "
            "age or horizon changes won't move it.")


def wealth_caption_text(projection_method: str, simulations: int = 5000) -> str:
    """Caption under a card's Projected Wealth title; both methods end the same way so the cards stay aligned."""
    if projection_method == "simple_percentiles":
        body = "Simple percentiles: steady growth at the expected and 25th/75th-percentile returns (no simulation)."
    else:
        body = f"Monte Carlo: {simulations:,} simulated paths; bands span the 10th–90th and 25th–75th percentiles."
    return f"{body} Same scale in all cards."


def wealth_caption(resp: PortfolioResponse, method: str) -> str:
    proj = resp.recommendation(method).projection
    return wealth_caption_text(proj.method, proj.simulations or 5000)


def summary_header(resp: PortfolioResponse, method: str) -> str:
    """Title and KPI tiles at the top of a portfolio's summary card."""
    return _card(resp, method)


def profile_chips(resp: PortfolioResponse) -> str:
    p = resp.profile
    target = resp.rule_based.projection.target_amount
    risk = (f"Risk <b>{escape(p.risk_tolerance_input)}</b>" if p.effective_risk_tolerance == p.risk_tolerance
            else f"Risk <b>{p.risk_tolerance:g} → {p.effective_risk_tolerance:g}</b>")
    chips = [
        f"👤 Age <b>{p.age}</b>",
        f"🎯 {escape(p.goal_label)} · <b>${target:,.0f}</b>" if target else f"🎯 {escape(p.goal_label)}",
        f"⏳ <b>{p.horizon_years}</b> yr horizon",
        (f"📈 {escape(p.glide_path)} glide path · <b>{p.equity_target:.0%}</b> equity"
         if p.equity_target is not None else f"📈 {escape(p.glide_path)} glide path"),
        (f"💼 Human capital <b>{money(p.human_capital)}</b>" if p.human_capital is not None else ""),
        f"{PROJECTION_ICONS.get(p.projection_method, '📊')} <b>{escape(p.projection_method)}</b> projection",
        f"🎚️ {risk} · {escape(p.risk_band)}",
        f"📅 Data as of <b>{escape(resp.market_data.data_as_of or '—')}</b>",
    ]
    title = escape(p.horizon_adjustment)
    return ("<div class='chips'>" + "".join(f"<span class='chip' title='{title}'>{c}</span>" for c in chips if c)
            + "</div>")


def status_panel(errors: dict[str, str] | None = None, warnings: list[str] | None = None,
                 message: str | None = None) -> str:
    if errors:
        items = "".join(f"<li>{escape(msg)}</li>" for msg in errors.values())
        return (f"<div class='status error'><div class='title'>⛔ Please fix {len(errors)} input"
                f"{'s' if len(errors) != 1 else ''}</div><ul>{items}</ul>"
                "<div>Charts show the last valid inputs.</div></div>")
    if message:
        return f"<div class='status error'><div class='title'>⛔ Something went wrong</div>{escape(message)}</div>"
    if warnings:
        items = "".join(f"<li>{escape(w)}</li>" for w in warnings)
        return f"<div class='status warn'><div class='title'>⚠️ Worth a second look</div><ul>{items}</ul></div>"
    return "<div class='status ok'><span class='title'>✅ Inputs look good</span> · dashboard is up to date</div>"


def asset_dot(asset_class: str) -> str:
    """Color swatch matching the asset class's slice in the allocation donut."""
    color = ASSET_CLASS_COLORS.get(asset_class, "#CBD5E1")
    return f"<span class='asset-dot' style='background-color:{color}' title='{escape(asset_class)}'></span>"


def holding_amount(amount: float) -> str:
    """Whole dollars, switching to $1.57M style from $1M so the narrow Amount column never truncates."""
    return f"${amount / 1e6:,.2f}M" if amount >= 999_995 else f"${amount:,.0f}"


def holdings_table(resp: PortfolioResponse, method: str) -> pd.DataFrame:
    rec = resp.recommendation(method)
    return pd.DataFrame(
        [{" ": asset_dot(h.asset_class), "Ticker": h.ticker,
          "Asset class": ASSET_CLASS_SHORT.get(h.asset_class, h.asset_class),
          "Exp. ret.": f"{h.expected_return:.1%}" if h.expected_return is not None else "—",
          "Weight": f"{h.weight:.1%}", "Amount": holding_amount(h.amount)}
         for h in rec.holdings],
        columns=HOLDINGS_COLUMNS,
    )


def comparison_header() -> str:
    """Title for the container whose charts plot all portfolios together."""
    vs = "<span class='vs'>vs</span>"
    title = vs.join(f"<span style='color:{METHOD_COLORS[m]}'>{METHOD_LABELS[m]}</span>" for m in METHODS)
    legend = " · ".join(f"<span class='swatch' style='background:{METHOD_COLORS[m]}'></span>{METHOD_TITLES[m]}"
                        for m in METHODS)
    bench = METHOD_COLORS["benchmark"]
    return (
        "<div class='comparison-head'>"
        f"<div class='title'>{title}</div>"
        f"<div class='sub'>All three portfolios are plotted together in each chart: {legend} · "
        f"<span class='swatch dotted' style='color:{bench}'></span>{METHOD_TITLES['benchmark']} benchmark</div>"
        "</div>"
    )


def research_insight_card(resp: PortfolioResponse) -> str:
    """Equity share of each portfolio and popular rule, plus where and why the research model disagrees."""
    insight = resp.research_insight
    bars = []
    for c in insight.comparisons:
        main = c.key == "research_informed"
        popular = c.key not in METHOD_COLORS
        color = POPULAR_RULE_COLOR if popular else METHOD_COLORS[c.key]
        css = "main" if main else "popular" if popular else ""
        bars.append(
            f"<div class='eq-bar {css}' data-key='{c.key}'><span class='lbl' title='{escape(c.label)}'>"
            f"{escape(c.label)}</span><span class='track'><span class='fill' style='display:block;"
            f"width:{max(c.equity, 0) * 100:.1f}%;background:{color}'></span></span>"
            f"<span class='val'>{c.equity:.0%}</span></div>"
        )
    points = "".join(f"<li>{escape(point)}</li>" for point in insight.points)
    sources = " · ".join(escape(s) for s in insight.sources)
    return (
        "<div class='insight'>"
        "<div class='head'><div class='title'>Research vs. Popular Wisdom</div>"
        f"<div class='sub'>{escape(insight.headline)}</div></div>"
        f"<div><div class='bars-title'>Stock allocation at age {resp.request.age}</div>{''.join(bars)}"
        "<div class='legend-note'>Gray bars are popular rules of thumb: 100 − age, and a typical target-date fund "
        "glide path (as modeled by Duarte et al., 2022).</div></div>"
        f"<div><div class='bars-title'>Where the research-informed model disagrees, and why</div>"
        f"<ul class='points'>{points}</ul><div class='sources'>Sources: {sources}</div></div>"
        "</div>"
    )


def backtest_caption(resp: PortfolioResponse) -> str:
    bt = resp.backtest
    growth = " · ".join(f"{m.label} {m.cagr:.1%}/yr" for m in bt.metrics.values())
    return (f"{bt.start} → {bt.end} · {bt.rebalance.capitalize()} rebalancing · "
            f"{_money_short(bt.initial_value)} initial (see notes)  \nGrowth: {growth}")


def notes_markdown(resp: PortfolioResponse) -> str:
    md = resp.market_data
    funds = {h.ticker: h.name for m in METHODS for h in resp.recommendation(m).holdings}
    sections = [
        "#### Recommendation notes",
        *[f"- {n}" for n in resp.notes],
        "#### Funds in these portfolios",
        *[f"- **{t}**: {escape(name)}" for t, name in sorted(funds.items())],
        "#### Backtest",
        *[f"- {n}" for n in resp.backtest.notes],
        "#### Assumptions",
        f"- Expected returns and risk are estimated from {md.observations} {md.frequency} returns "
        f"({md.estimation_start} to {md.estimation_end}); the risk-free rate is {md.risk_free_rate:.1%}.",
        f"- Your {resp.profile.horizon_years}-year horizon sets the effective risk level: "
        f"{resp.profile.horizon_adjustment}.",
        (f"- Monte Carlo projections use {resp.rule_based.projection.simulations:,} simulated month-by-month return "
         "paths, with monthly contributions added after each month's growth; goal odds are the share of paths that "
         "reach the target."
         if resp.rule_based.projection.method == "monte_carlo" else
         "- Simple-percentile projections grow the initial investment and monthly contributions at constant "
         "percentile returns of the lognormal return distribution for each horizon (no simulation); goal odds are "
         "the chance the horizon's annualized return beats the return needed to reach the target."),
        "- Mean-variance weights are estimated on history that overlaps the backtest, so its backtest is in-sample.",
        "- The research-informed equity share uses long-run research assumptions (Practical Finance baseline: "
        "4% log equity premium, 18.5% stock volatility, 2% real risk-free rate), not the historical estimates above; "
        "risk tolerance 1–10 maps to relative risk aversion 10–4.",
        "- For education only; this is not investment advice.",
    ]
    return "\n".join(sections)


def method_label(method: str) -> str:
    return METHOD_LABELS[method]
