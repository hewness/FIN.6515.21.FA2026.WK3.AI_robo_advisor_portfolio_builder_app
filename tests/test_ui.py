import json

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import pytest

from portfolio_builder.ui import Dashboard, build_demo, charts, components
from portfolio_builder.ui.app import (
    HEAD,
    HOLDINGS_TABLE_HEIGHT,
    INPUT_FIELDS,
    MAX_HOLDINGS,
    MONEY_FIELDS,
    OUTPUT_KEYS,
    default_values,
    preset_for,
    sidebar_tooltips,
)
from portfolio_builder.ui.formatting import format_money, parse_money
from portfolio_builder.ui.theme import ASSET_CLASS_COLORS, FRONTIER_COLOR, METHOD_COLORS, TARGET_COLOR


@pytest.fixture(scope="module")
def response(portfolio_service):
    return portfolio_service.build_portfolio(default_values())


def trace_names(fig: go.Figure) -> list[str]:
    return [t.name for t in fig.data]


def test_allocation_donut_per_portfolio(response):
    figs = {m: charts.allocation_donut(response, m) for m in ("rule_based", "mean_variance")}
    for method, fig in figs.items():
        assert [t.type for t in fig.data] == ["pie"]
        assert sum(fig.data[0].values) == pytest.approx(1.0)
        assert len(fig.data[0].labels) == len(response.recommendation(method).asset_classes)
    assert "Cash" in figs["rule_based"].data[0].labels  # rule-based holds cash
    # same asset class gets the same color in both donuts
    colors = [dict(zip(f.data[0].labels, f.data[0].marker.colors)) for f in figs.values()]
    for label in set(colors[0]) & set(colors[1]):
        assert colors[0][label] == colors[1][label]


def test_risk_return_scatter(response):
    names = trace_names(charts.risk_return_scatter(response))
    assert names == ["Efficient frontier", "Asset classes", "Max Sharpe",
                     "Rule-based Lifecycle Portfolio", "Mean-variance Optimized Portfolio"]
    fig = charts.risk_return_scatter(response)
    assert len(fig.data[1].x) == 7  # 6 asset classes + cash
    frontier = fig.data[0]
    assert frontier.line.dash == "dash" and frontier.line.color == FRONTIER_COLOR
    assert fig.data[3].y[0] == response.rule_based.expected_return


def test_wealth_projection_per_portfolio_on_shared_scale(response):
    y_max = charts.wealth_y_max(response)
    for method in ("rule_based", "mean_variance"):
        fig = charts.wealth_projection(response, method, y_max)
        names = set(trace_names(fig))
        assert {"Expected (mean)", "Optimistic (75th pct)", "Pessimistic (25th pct)", "Total contributed"} <= names
        expected = next(t for t in fig.data if t.name == "Expected (mean)")
        assert len(expected.x) == response.request.horizon_years + 1
        assert expected.line.color == METHOD_COLORS[method]
        assert tuple(fig.layout.yaxis.range) == (0, y_max)
        assert any("Goal" in (a.text or "") for a in fig.layout.annotations)
    highest = max(p.p75 for m in ("rule_based", "mean_variance") for p in response.recommendation(m).projection.points)
    assert y_max >= highest


def test_backtest_chart(response):
    fig = charts.backtest_chart(response)
    assert trace_names(fig)[::2] == ["Rule-based Lifecycle Portfolio", "Mean-variance Optimized Portfolio", "S&P 500 (SPY)"]
    assert [t.line.color for t in fig.data[::2]] == [METHOD_COLORS[k] for k in ("rule_based", "mean_variance", "benchmark")]
    assert len(fig.data) == 6 and len(fig.data[0].x) == len(response.backtest.points)


def test_components(response):
    for method, title in (("rule_based", "Rule-based Lifecycle Portfolio"),
                          ("mean_variance", "Mean-variance Optimized Portfolio")):
        html = components.summary_header(response, method)
        assert title in html
        for label in ("Exp. return", "Volatility", "Sharpe", "Max drawdown", "Goal odds"):
            assert html.count(label) == 1

    chips = components.profile_chips(response)
    assert "Age <b>40</b>" in chips and "$1,500,000" in chips

    table = components.holdings_table(response, "rule_based")
    assert list(table.columns) == [" ", "Ticker", "Asset class", "Weight", "Amount"]
    assert table["Weight"].str.endswith("%").all() and table["Amount"].str.startswith("$").all()
    # indicator color matches the asset class's donut slice color
    donut = charts.allocation_donut(response, "rule_based").data[0]
    slice_colors = dict(zip(donut.labels, donut.marker.colors))
    for _, row in table.iterrows():
        assert f"background-color:{slice_colors[row['Asset class']]}" in row[" "]

    header = components.comparison_header()
    assert "Rule-based Lifecycle Portfolio" in header and "Mean-variance Optimized Portfolio" in header
    assert METHOD_COLORS["rule_based"] in header and METHOD_COLORS["mean_variance"] in header

    notes = components.notes_markdown(response)
    assert "Assumptions" in notes and "Funds in these portfolios" in notes and "**VTI**: VTI Fund" in notes
    assert "rebalancing" in components.backtest_caption(response)


def test_status_panel():
    assert "status ok" in components.status_panel()
    warn = components.status_panel(warnings=["Target <b>far</b>"])
    assert "status warn" in warn and "&lt;b&gt;" in warn  # escaped
    err = components.status_panel(errors={"age": "Age must be a number between 18 and 80"})
    assert "status error" in err and "Please fix 1 input" in err and "18 and 80" in err


def test_presets():
    assert preset_for(3.0) == "Conservative" and preset_for(5.5) == "Moderate" and preset_for(8) == "Aggressive"
    assert preset_for(6.5) == "Custom" and preset_for(None) == "Custom"


def test_build_demo_constructs(portfolio_service):
    demo = build_demo(portfolio_service)
    assert isinstance(demo, gr.Blocks)


def test_dashboard_update_valid_and_invalid(portfolio_service):
    dashboard = Dashboard(portfolio_service)
    values = default_values()
    out = dict(zip(OUTPUT_KEYS, dashboard.update(*[values[f] for f in INPUT_FIELDS])))
    assert len(out) == len(OUTPUT_KEYS) == 14
    assert "status" in out["status"]
    for method in ("rule_based", "mean_variance"):
        assert "tiles" in out[f"header_{method}"]
        assert isinstance(out[f"donut_{method}"], go.Figure)
        assert isinstance(out[f"table_{method}"], pd.DataFrame)
        assert isinstance(out[f"wealth_{method}"], go.Figure)
    assert all(isinstance(out[k], go.Figure) for k in ("scatter", "backtest"))

    as_text = {**values, **{f: format_money(values[f]) for f in MONEY_FIELDS}}
    assert as_text["initial_investment"] == "$50,000"
    text_out = dict(zip(OUTPUT_KEYS, dashboard.update(*[as_text[f] for f in INPUT_FIELDS])))
    assert "status ok" in text_out["status"]
    assert "$50K" in text_out["donut_rule_based"].data[0].title.text  # "$50,000" text parsed as 50,000

    typo = {**as_text, "initial_investment": "fifty thousand"}
    out = dashboard.update(*[typo[f] for f in INPUT_FIELDS])
    assert "Initial investment ($) must be a number between 1,000 and 10,000,000" in out[0]

    bad = {**values, "age": 95, "initial_investment": "$5"}
    out = dashboard.update(*[bad[f] for f in INPUT_FIELDS])
    assert len(out) == len(OUTPUT_KEYS)
    assert "Please fix 2 inputs" in out[0]
    assert all(o == gr.skip() for o in out[1:])


def test_asset_palette_does_not_reuse_portfolio_colors():
    assert not set(ASSET_CLASS_COLORS.values()) & set(METHOD_COLORS.values())
    assert FRONTIER_COLOR not in {*ASSET_CLASS_COLORS.values(), *METHOD_COLORS.values(), TARGET_COLOR}
    assert len(set(ASSET_CLASS_COLORS.values())) == len(ASSET_CLASS_COLORS)


def test_hover_script_targets_card_donut_and_table():
    for needle in (".portfolio-card", ".donut-plot", "plotly_hover", "plotly_unhover", ".virtual-row",
                   "Asset class", "row-active", "row-dim"):
        assert needle in HEAD


@pytest.mark.parametrize("text,expected", [
    ("$50,000", 50_000.0), ("50000", 50_000.0), (" $1,234,567 ", 1_234_567.0), ("50k", 50_000.0),
    ("2.5K", 2_500.0), ("1.2m", 1_200_000.0), ("$0", 0.0), (1500, 1_500.0), (99.5, 99.5),
    ("", None), (None, None), ("   ", None), ("fifty", "fifty"), ("$5-", "$5-"), ("1e6", "1e6"),
])
def test_parse_money(text, expected):
    assert parse_money(text) == expected


@pytest.mark.parametrize("value,expected", [
    (50_000, "$50,000"), ("1234567", "$1,234,567"), ("50k", "$50,000"), ("$1,000", "$1,000"),
    (0, "$0"), (999.6, "$1,000"), ("", ""), (None, ""), ("abc", "abc"),
])
def test_format_money(value, expected):
    assert format_money(value) == expected


def test_money_inputs_are_formatted_textboxes(portfolio_service):
    demo = build_demo(portfolio_service)
    boxes = {b.elem_id: b for b in demo.blocks.values() if getattr(b, "elem_id", None) in ("in-initial", "in-monthly", "in-target")}
    assert set(boxes) == {"in-initial", "in-monthly", "in-target"}
    assert all(isinstance(b, gr.Textbox) and "money-input" in (b.elem_classes or []) for b in boxes.values())
    assert boxes["in-initial"].value == "$50,000" and boxes["in-monthly"].value == "$1,000"
    assert boxes["in-target"].value == "$1,500,000"


def test_sidebar_help_moves_to_label_tooltips(portfolio_service):
    demo = build_demo(portfolio_service)
    sidebar_ids = {b.elem_id for b in demo.blocks.values() if str(getattr(b, "elem_id", "") or "").startswith("in-")}
    tips = sidebar_tooltips()
    assert sidebar_ids == set(tips)
    assert all(not getattr(b, "info", None) for b in demo.blocks.values() if getattr(b, "elem_id", None) in tips)
    assert json.dumps(tips["in-initial"]) in HEAD
    for needle in ("has-tip", "block-info", "money-input", "toLocaleString"):
        assert needle in HEAD


def test_holdings_table_fits_largest_portfolio_without_scrolling(portfolio_service):
    assert MAX_HOLDINGS == 7  # 6 funds + cash
    assert HOLDINGS_TABLE_HEIGHT >= 45 + 36 * MAX_HOLDINGS
    demo = build_demo(portfolio_service)
    tables = [b for b in demo.blocks.values() if isinstance(b, gr.Dataframe)]
    assert len(tables) == 2 and all(t.max_height == HOLDINGS_TABLE_HEIGHT for t in tables)
