import json

import pytest

from portfolio_builder.market_data import MarketDataService, ParquetCache
from portfolio_builder.optimization import PortfolioOptimizationEngine
from portfolio_builder.service import (
    InputValidationError,
    PortfolioResponse,
    PortfolioService,
    PortfolioServiceError,
)
from portfolio_builder.service import portfolio_service as ps

from .conftest import FakeConnector

REQUEST = {
    "risk_tolerance": "moderate",
    "horizon_years": 20,
    "initial_investment": 250_000,
    "monthly_contribution": 1_000,
    "goal": "retirement",
    "age": 45,
}


@pytest.fixture(scope="module")
def response(portfolio_service) -> PortfolioResponse:
    return portfolio_service.build_portfolio(REQUEST)


def test_profile_and_market_summary(response):
    p = response.profile
    assert (p.risk_tolerance_input, p.risk_tolerance, p.effective_risk_tolerance) == ("Moderate", 5.5, 6.5)
    assert p.risk_band == "Moderate" and p.goal_label == "Retirement"
    assert "+1" in p.horizon_adjustment
    md = response.market_data
    assert md.observations >= 24 and md.risk_free_rate == 0.02 and md.data_as_of is not None


@pytest.mark.parametrize("method", ["rule_based", "mean_variance"])
def test_holdings_weights_and_amounts(response, method):
    rec = response.recommendation(method)
    assert rec.method == method
    assert sum(h.weight for h in rec.holdings) == pytest.approx(1.0)
    assert all(h.weight > 0 for h in rec.holdings)
    assert round(sum(h.amount for h in rec.holdings), 2) == 250_000
    assert sum(a.amount for a in rec.asset_classes) == pytest.approx(250_000, abs=0.01)
    assert [h.weight for h in rec.holdings] == sorted((h.weight for h in rec.holdings), reverse=True)
    assert rec.projection.points[-1].year == 20
    assert rec.projection.total_contributed == 250_000 + 1_000 * 240


def test_rule_based_contents(response):
    rec = response.rule_based
    assert any(h.ticker == "CASH" and h.name == "Cash and Money Markets" for h in rec.holdings)
    assert any(h.name == "VTI Fund" for h in rec.holdings)  # fund names from cached info
    assert len(rec.holdings) == 7  # one fund per asset class + cash
    assert 0 < rec.details["equity_pct"] <= 1


def test_mean_variance_on_frontier(response):
    rec = response.mean_variance
    assert rec.frontier_position.on_frontier
    assert rec.details["target_volatility"] == pytest.approx(rec.volatility, abs=1e-5)
    frontier = response.efficient_frontier
    assert len(frontier.points) > 10
    assert {r.name for r in frontier.reference_portfolios} == {"min_volatility", "max_sharpe", "max_return"}
    assert frontier.client_point.volatility == rec.volatility
    assert frontier.rule_based_point.expected_return == response.rule_based.expected_return


def test_dataframe_helpers(response):
    assert list(response.holdings_frame("rule_based").columns) == ["ticker", "name", "asset_class", "role", "weight", "amount"]
    assert list(response.asset_class_frame("mean_variance").columns)[:2] == ["asset_class", "role"]
    assert len(response.projection_frame("mean_variance")) == 21
    frontier = response.frontier_frame()
    assert {"expected_return", "volatility", "sharpe_ratio"} <= set(frontier.columns) and len(frontier) > 10
    comparison = response.comparison_frame()
    assert list(comparison["method"]) == ["rule_based", "mean_variance"]
    with pytest.raises(ValueError):
        response.holdings_frame("black_litterman")


def test_json_serializable(response):
    data = json.loads(response.model_dump_json())
    assert data["request"]["goal"] == "retirement"
    assert data["request"]["risk_tolerance"] == "moderate"
    assert json.dumps(response.to_dict())
    assert PortfolioResponse.model_validate(data).profile.effective_risk_tolerance == 6.5


def test_notes(response):
    text = " ".join(response.notes)
    assert "Retirement" in text
    assert "horizon changed the risk level" in text
    assert "not guarantees" in text


def test_short_horizon_home_purchase(portfolio_service):
    res = portfolio_service.build_portfolio({**REQUEST, "risk_tolerance": 9, "horizon_years": 2, "goal": "Home purchase"})
    assert res.profile.effective_risk_tolerance == 3.0
    assert res.profile.risk_band == "Conservative"
    assert any("short-term TIPS" in n and "down payment" in n for n in res.notes)


def test_invalid_input_field_errors(portfolio_service):
    with pytest.raises(InputValidationError) as info:
        portfolio_service.build_portfolio({**REQUEST, "age": 81, "initial_investment": 10, "goal": "boat", "extra": 1})
    errors = info.value.field_errors
    assert set(errors) == {"age", "initial_investment", "goal", "extra"}
    assert errors["age"] == "Age must be a number between 18 and 80"
    assert "Home purchase" in errors["goal"]


def test_engine_failure_becomes_service_error(tmp_path):
    market = MarketDataService(FakeConnector(), ParquetCache(tmp_path / "short"))  # too little history
    service = PortfolioService(engine=PortfolioOptimizationEngine(market_data=market))
    with pytest.raises(PortfolioServiceError, match="Could not build a portfolio"):
        service.build_portfolio(REQUEST)


def test_recommend_portfolio_uses_singleton(monkeypatch, portfolio_service):
    monkeypatch.setattr(ps, "_service", None)
    monkeypatch.setattr(ps, "PortfolioService", lambda: portfolio_service)
    first = ps.get_portfolio_service()
    assert ps.get_portfolio_service() is first is portfolio_service

    res = ps.recommend_portfolio(8, 10, 5_000, 0, "education", 30)
    assert res.profile.effective_risk_tolerance == 8.0
    assert res.request.goal.value == "education"
    assert first.get_form_options()["age"]["maximum"] == 80


def test_goal_probability_and_drawdown(response):
    for rec in (response.rule_based, response.mean_variance):
        assert rec.projection.target_amount == 1_500_000  # retirement default
        assert 0.0 <= rec.probability_of_meeting_target <= 1.0
        assert rec.max_drawdown is not None and rec.max_drawdown <= 0
        p = rec.projection.points[-1]
        assert p.p10 <= p.p25 <= p.p50 <= p.p75 <= p.p90


def test_backtest_in_response(response):
    bt = response.backtest
    assert set(bt.metrics) == {"rule_based", "mean_variance", "benchmark"}
    assert bt.metrics["benchmark"].label == "S&P 500 (SPY)" and bt.benchmark == "SPY"
    assert bt.initial_value == 250_000 and bt.rebalance == "quarterly" and bt.years_requested == 15
    assert bt.points[0].rule_based == 250_000 and bt.points[-1].date == bt.end
    assert any("years of history" in n for n in bt.notes)  # test data covers ~5 years
    assert response.rule_based.max_drawdown == bt.metrics["rule_based"].max_drawdown
    frame = response.backtest_frame()
    assert list(frame.columns[:4]) == ["date", "rule_based", "mean_variance", "benchmark"]
    assert str(frame["date"].dtype).startswith("datetime64")


def test_asset_class_points(response):
    points = {p.asset_class: p for p in response.efficient_frontier.asset_class_points}
    assert len(points) == 7
    assert points["Cash and Money Markets"].volatility == 0
    assert points["Cash and Money Markets"].expected_return == response.market_data.risk_free_rate
    assert points["US large-cap stocks"].tickers == ["VTI"]


def test_explicit_target_and_warnings(portfolio_service):
    far = portfolio_service.build_portfolio({**REQUEST, "target_amount": 90_000_000})
    assert far.rule_based.probability_of_meeting_target == 0.0
    assert any("hard to reach" in w for w in far.warnings)

    met = portfolio_service.build_portfolio({**REQUEST, "target_amount": 100_000})
    assert met.rule_based.probability_of_meeting_target == 1.0
    assert any("already met" in w for w in met.warnings)

    long_home = portfolio_service.build_portfolio({**REQUEST, "goal": "home_purchase", "horizon_years": 25, "age": 30})
    assert any("unusually long" in w for w in long_home.warnings)

    late = portfolio_service.build_portfolio({**REQUEST, "age": 60, "horizon_years": 25})
    assert any("age 85" in w for w in late.warnings)

    capped = portfolio_service.build_portfolio({**REQUEST, "risk_tolerance": "aggressive", "horizon_years": 1})
    assert any("capped at 3" in w for w in capped.warnings)


def test_backtest_settings_and_empty_number(portfolio_service):
    res = portfolio_service.build_portfolio({**REQUEST, "backtest_years": 10, "rebalance": "monthly"})
    assert (res.backtest.years_requested, res.backtest.rebalance) == (10, "monthly")
    with pytest.raises(InputValidationError) as info:
        portfolio_service.build_portfolio({**REQUEST, "monthly_contribution": None, "backtest_years": 25})
    assert info.value.field_errors["monthly_contribution"] == "Monthly contribution ($) must be a number between 0 and 50,000"
    assert "10 and 20" in info.value.field_errors["backtest_years"]
