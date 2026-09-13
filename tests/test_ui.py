import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import pytest

from portfolio_builder.ui import Dashboard, build_demo, charts, components
from portfolio_builder.ui.app import INPUT_FIELDS, OUTPUT_KEYS, default_values, preset_for


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
    assert names == ["Efficient frontier", "Asset classes", "Max Sharpe", "Rule-based portfolio", "Mean-variance portfolio"]
    fig = charts.risk_return_scatter(response)
    assert len(fig.data[1].x) == 7  # 6 asset classes + cash
    assert fig.data[3].y[0] == response.rule_based.expected_return


def test_wealth_projection(response):
    fig = charts.wealth_projection(response)
    names = set(trace_names(fig))
    assert {"Expected (mean)", "Optimistic (75th pct)", "Pessimistic (25th pct)", "Total contributed"} <= names
    expected = [t for t in fig.data if t.name == "Expected (mean)"]
    assert len(expected) == 2 and len(expected[0].x) == response.request.horizon_years + 1
    assert any("Goal" in (a.text or "") for a in fig.layout.annotations)


def test_backtest_chart(response):
    fig = charts.backtest_chart(response)
    assert trace_names(fig)[::2] == ["Rule-based", "Mean-variance", "S&P 500 (SPY)"]
    assert len(fig.data) == 6 and len(fig.data[0].x) == len(response.backtest.points)


def test_components(response):
    for method, title in (("rule_based", "Rule-based lifecycle portfolio"),
                          ("mean_variance", "Mean-variance optimized portfolio")):
        html = components.summary_header(response, method)
        assert title in html
        for label in ("Exp. return", "Volatility", "Sharpe", "Max drawdown", "Goal odds"):
            assert html.count(label) == 1

    chips = components.profile_chips(response)
    assert "Age <b>40</b>" in chips and "$1,500,000" in chips

    table = components.holdings_table(response, "rule_based")
    assert list(table.columns) == ["Ticker", "Asset class", "Weight", "Amount"]
    assert table["Weight"].str.endswith("%").all() and table["Amount"].str.startswith("$").all()

    notes = components.notes_markdown(response)
    assert "Assumptions" in notes and "Funds in these portfolios" in notes and "**SPY**: SPY Fund" in notes
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
    assert len(out) == len(OUTPUT_KEYS) == 13
    assert "status" in out["status"]
    for method in ("rule_based", "mean_variance"):
        assert "tiles" in out[f"header_{method}"]
        assert isinstance(out[f"donut_{method}"], go.Figure)
        assert isinstance(out[f"table_{method}"], pd.DataFrame)
    assert all(isinstance(out[k], go.Figure) for k in ("scatter", "wealth", "backtest"))

    bad = {**values, "age": 95, "initial_investment": 5}
    out = dashboard.update(*[bad[f] for f in INPUT_FIELDS])
    assert len(out) == len(OUTPUT_KEYS)
    assert "Please fix 2 inputs" in out[0]
    assert all(o == gr.skip() for o in out[1:])
