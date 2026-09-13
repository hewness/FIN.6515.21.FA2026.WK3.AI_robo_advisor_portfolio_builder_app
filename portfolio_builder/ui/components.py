"""HTML snippets, tables and markdown for the dashboard."""

from __future__ import annotations

from html import escape

import pandas as pd

from ..service import PortfolioResponse
from ..service.schemas import PortfolioRecommendation
from .charts import _money_short
from .theme import ASSET_CLASS_SHORT, METHOD_COLORS, METHOD_LABELS

METHODS = ("rule_based", "mean_variance")
HOLDINGS_COLUMNS = ["Ticker", "Asset class", "Weight", "Amount"]


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
              f"Probability of reaching ${proj.target_amount or 0:,.0f} by age {end_age}: share of "
              f"{proj.simulations:,} simulated outcomes that end at or above the goal target.",
              _probability_class(prob)),
    ]
    color = METHOD_COLORS[method]
    return (
        f"<div class='card-head'><span class='name' style='color:{color}'>{escape(rec.title)}</span>"
        f"<span class='desc'>{escape(rec.description)}</span></div>"
        f"<div class='tiles'>{''.join(tiles)}</div>"
    )


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
        f"🎚️ {risk} · {escape(p.risk_band)}",
        f"📅 Data as of <b>{escape(resp.market_data.data_as_of or '—')}</b>",
    ]
    title = escape(p.horizon_adjustment)
    return "<div class='chips'>" + "".join(f"<span class='chip' title='{title}'>{c}</span>" for c in chips) + "</div>"


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


def holdings_table(resp: PortfolioResponse, method: str) -> pd.DataFrame:
    rec = resp.recommendation(method)
    return pd.DataFrame(
        [{"Ticker": h.ticker, "Asset class": ASSET_CLASS_SHORT.get(h.asset_class, h.asset_class),
          "Weight": f"{h.weight:.1%}", "Amount": f"${h.amount:,.0f}"}
         for h in rec.holdings],
        columns=HOLDINGS_COLUMNS,
    )


def backtest_caption(resp: PortfolioResponse) -> str:
    bt = resp.backtest
    parts = [f"{bt.start} → {bt.end} ({bt.years_covered:.1f} years)", f"{bt.rebalance} rebalancing",
             f"${bt.initial_value:,.0f} initial investment, no contributions"]
    cagr = " · ".join(f"{m.label} CAGR {m.cagr:.1%}" for m in bt.metrics.values())
    proxy = " · uses sibling-fund proxies before some funds existed" if bt.proxies_used else ""
    return f"{' · '.join(parts)}{proxy}  \n{cagr}"


def notes_markdown(resp: PortfolioResponse) -> str:
    md = resp.market_data
    funds = {h.ticker: h.name for rec in (resp.rule_based, resp.mean_variance) for h in rec.holdings}
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
        f"- Projections use {resp.rule_based.projection.simulations:,} simulated return paths with monthly "
        "contributions added after each month's growth.",
        "- Mean-variance weights are estimated on history that overlaps the backtest, so its backtest is in-sample.",
        "- For education only; this is not investment advice.",
    ]
    return "\n".join(sections)


def method_label(method: str) -> str:
    return METHOD_LABELS[method]
