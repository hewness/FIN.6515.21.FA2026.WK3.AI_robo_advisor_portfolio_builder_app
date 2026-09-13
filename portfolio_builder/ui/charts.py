"""Plotly figures for the dashboard; each is a pure function of a ``PortfolioResponse``."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..service import PortfolioResponse
from ..universe import get_asset_classes
from .theme import (
    ASSET_CLASS_COLORS,
    ASSET_CLASS_SHORT,
    BENCHMARK_COLOR,
    GRID_COLOR,
    METHOD_COLORS,
    METHOD_LABELS,
    MUTED_TEXT,
    TARGET_COLOR,
)

METHODS = ("rule_based", "mean_variance")
ASSET_CLASS_ORDER = [ac.name for ac in get_asset_classes()]


def _style(fig: go.Figure, height: int, legend_y: float = -0.18) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", size=12, color=MUTED_TEXT),
        legend=dict(orientation="h", yanchor="top", y=legend_y, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=12, font_family="Inter, system-ui, sans-serif"),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    return fig


LABEL_POSITIONS = {
    "Cash and Money Markets": "top right",
    "Treasury inflation-protected securities": "bottom left",
    "US Aggregate bonds": "bottom right",
    "US large-cap stocks": "top left",
    "International developed stocks": "top left",
    "Emerging market stocks": "bottom center",
    "Real estate (REITs)": "top center",
}


def _money_short(value: float) -> str:
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"${value / size:,.1f}".rstrip("0").rstrip(".") + suffix
    return f"${value:,.0f}"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def allocation_donut(resp: PortfolioResponse, method: str) -> go.Figure:
    """Donut of one portfolio's asset-class weights; colors are fixed per asset class across portfolios."""
    rec = resp.recommendation(method)
    by_class = {a.asset_class: a for a in rec.asset_classes}
    classes = [c for c in ASSET_CLASS_ORDER if c in by_class]
    fig = go.Figure(
        go.Pie(
            labels=[ASSET_CLASS_SHORT.get(c, c) for c in classes],
            values=[by_class[c].weight for c in classes],
            customdata=[[c, by_class[c].amount, by_class[c].role] for c in classes],
            marker=dict(colors=[ASSET_CLASS_COLORS.get(c, "#CBD5E1") for c in classes],
                        line=dict(color="rgba(255,255,255,0.9)", width=2)),
            hole=0.6,
            sort=False,
            direction="clockwise",
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            name=METHOD_LABELS[method],
            title=dict(text=f"<b>{_money_short(resp.request.initial_investment)}</b>", position="middle center",
                       font=dict(size=15)),
            hovertemplate="<b>%{customdata[0][0]}</b><br>%{percent} · $%{customdata[0][1]:,.0f}"
                          "<br>%{customdata[0][2]}<extra></extra>",
        )
    )
    fig.update_traces(domain=dict(x=[0.05, 0.95], y=[0.3, 1.0]))  # fixed donut size whatever the legend length
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")
    _style(fig, height=340)
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4), font=dict(size=11),
        legend=dict(orientation="h", yanchor="top", y=0.26, xanchor="center", x=0.5, font=dict(size=10.5),
                    entrywidth=0.5, entrywidthmode="fraction"),
    )
    return fig


def risk_return_scatter(resp: PortfolioResponse) -> go.Figure:
    """Asset classes, cash, the efficient frontier and both recommended portfolios in risk/return space."""
    ef = resp.efficient_frontier
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p.volatility for p in ef.points], y=[p.expected_return for p in ef.points],
        mode="lines", name="Efficient frontier", line=dict(color=_rgba(METHOD_COLORS["mean_variance"], 0.55), width=3),
        customdata=[p.sharpe_ratio for p in ef.points],
        hovertemplate="Frontier<br>Return %{y:.2%} · Risk %{x:.2%}<br>Sharpe %{customdata:.2f}<extra></extra>",
    ))
    points = ef.asset_class_points
    fig.add_trace(go.Scatter(
        x=[a.volatility for a in points], y=[a.expected_return for a in points],
        mode="markers+text", name="Asset classes",
        text=[ASSET_CLASS_SHORT.get(a.asset_class, a.asset_class) for a in points],
        textposition=[LABEL_POSITIONS.get(a.asset_class, "top center") for a in points], textfont=dict(size=11),
        marker=dict(size=11, color=[ASSET_CLASS_COLORS.get(a.asset_class, "#94A3B8") for a in points],
                    line=dict(color="white", width=1.5)),
        customdata=[[a.asset_class, ", ".join(a.tickers)] for a in points],
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Return %{y:.2%} · Risk %{x:.2%}<extra></extra>",
    ))
    max_sharpe = next((r for r in ef.reference_portfolios if r.name == "max_sharpe"), None)
    if max_sharpe is not None:
        fig.add_trace(go.Scatter(
            x=[max_sharpe.volatility], y=[max_sharpe.expected_return], mode="markers", name="Max Sharpe",
            marker=dict(symbol="diamond", size=12, color="#FBBF24", line=dict(color="#92400E", width=1)),
            hovertemplate=f"<b>Max Sharpe</b> ({max_sharpe.sharpe_ratio:.2f})<br>Return %{{y:.2%}} · Risk %{{x:.2%}}<extra></extra>",
        ))
    for method in METHODS:
        rec = resp.recommendation(method)
        fig.add_trace(go.Scatter(
            x=[rec.volatility], y=[rec.expected_return], mode="markers", name=f"{METHOD_LABELS[method]} portfolio",
            marker=dict(symbol="star", size=20, color=METHOD_COLORS[method], line=dict(color="white", width=1.5)),
            hovertemplate=(f"<b>{METHOD_LABELS[method]} portfolio</b><br>Return %{{y:.2%}} · Risk %{{x:.2%}}"
                           f"<br>Sharpe {rec.sharpe_ratio:.2f}<extra></extra>"),
        ))
    xs = [p.volatility for p in ef.points] + [a.volatility for a in points]
    ys = [p.expected_return for p in ef.points] + [a.expected_return for a in points]
    fig.update_xaxes(title_text="Risk (annual volatility)", tickformat=".0%", range=[-0.01, max(xs) * 1.12])
    fig.update_yaxes(title_text="Expected annual return", tickformat=".0%",
                     range=[min(0.0, min(ys)) - 0.01, max(ys) * 1.12])
    return _style(fig, height=360, legend_y=-0.2)


def wealth_projection(resp: PortfolioResponse) -> go.Figure:
    """Projected value over the horizon: expected, optimistic (p75) and pessimistic (p25) per portfolio."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[METHOD_LABELS[m] for m in METHODS])
    target = resp.rule_based.projection.target_amount
    for col, method in enumerate(METHODS, start=1):
        proj = resp.recommendation(method).projection
        color = METHOD_COLORS[method]
        years = [p.year for p in proj.points]
        first = col == 1
        fig.add_trace(go.Scatter(x=years, y=[p.p75 for p in proj.points], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=1, col=col)
        fig.add_trace(go.Scatter(x=years, y=[p.p25 for p in proj.points], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_rgba(color, 0.14), name="25th–75th percentile range",
                                 legendgroup="band", showlegend=first, hoverinfo="skip"), row=1, col=col)
        for key, label, dash, width in (("p75", "Optimistic (75th pct)", "dash", 2),
                                        ("expected", "Expected (mean)", "solid", 3),
                                        ("p25", "Pessimistic (25th pct)", "dot", 2)):
            fig.add_trace(go.Scatter(
                x=years, y=[getattr(p, key) for p in proj.points], mode="lines", name=label,
                line=dict(color=color, dash=dash, width=width), legendgroup=key, showlegend=first,
                hovertemplate=f"{METHOD_LABELS[method]} · {label}<br>Year %{{x}}: $%{{y:,.0f}}<extra></extra>",
            ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=years, y=[p.total_contributed for p in proj.points], mode="lines", name="Total contributed",
            line=dict(color=BENCHMARK_COLOR, dash="dashdot", width=1.5), legendgroup="contrib", showlegend=first,
            hovertemplate="Contributed by year %{x}: $%{y:,.0f}<extra></extra>",
        ), row=1, col=col)
        if target:
            prob = proj.probability_of_meeting_target
            fig.add_hline(
                y=target, row=1, col=col, line=dict(color=TARGET_COLOR, dash="dot", width=2),
                annotation_text=f"Goal ${target:,.0f}" + (f" · {prob:.0%} chance" if prob is not None else ""),
                annotation_position="top left", annotation_font=dict(color=TARGET_COLOR, size=11),
            )
    fig.update_xaxes(title_text="Years from today")
    fig.update_yaxes(tickprefix="$", tickformat="~s", row=1, col=1)
    fig.update_layout(hovermode="x unified")
    return _style(fig, height=400, legend_y=-0.2)


def backtest_chart(resp: PortfolioResponse) -> go.Figure:
    """Growth of the initial investment and drawdowns versus the S&P 500."""
    bt = resp.backtest
    dates = [p.date for p in bt.points]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    for key in ("rule_based", "mean_variance", "benchmark"):
        color = METHOD_COLORS[key]
        label = bt.metrics[key].label if key in bt.metrics else METHOD_LABELS[key]
        width = 2 if key == "benchmark" else 2.5
        dash = "dot" if key == "benchmark" else "solid"
        fig.add_trace(go.Scatter(
            x=dates, y=[getattr(p, key) for p in bt.points], mode="lines", name=label, legendgroup=key,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"{label}: $%{{y:,.0f}}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=[getattr(p, f"{key}_drawdown") for p in bt.points], mode="lines", name=f"{label} drawdown",
            legendgroup=key, showlegend=False, line=dict(color=color, width=1),
            fill="tozeroy", fillcolor=_rgba(color, 0.10 if key == "benchmark" else 0.16),
            hovertemplate=f"{label} drawdown: %{{y:.1%}}<extra></extra>",
        ), row=2, col=1)
    fig.update_yaxes(title_text="Portfolio value", tickprefix="$", tickformat="~s", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return _style(fig, height=460, legend_y=-0.1)
