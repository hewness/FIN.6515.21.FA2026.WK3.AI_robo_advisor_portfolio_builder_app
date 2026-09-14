import pandas as pd
import pytest

from portfolio_builder.market_data import MarketDataService, ParquetCache
from portfolio_builder.optimization import (
    AllocationResult,
    AllocationStrategy,
    InvestorProfile,
    OptimizationError,
    PortfolioOptimizationEngine,
    portfolio_metrics,
)
from portfolio_builder.universe import get_tickers

from .conftest import FakeConnector, LongHistoryConnector

PROFILE = InvestorProfile(age=40, risk_tolerance=6, annual_income=80_000, financial_wealth=100_000)


class EqualWeight(AllocationStrategy):
    name = "equal_weight"

    def allocate(self, profile, inputs):
        w = pd.Series(1.0 / len(inputs.tickers), index=inputs.tickers)
        return AllocationResult(self.name, profile, w, portfolio_metrics(w, inputs))


@pytest.fixture
def engine(tmp_path):
    service = MarketDataService(LongHistoryConnector(), ParquetCache(tmp_path / "cache"))
    return PortfolioOptimizationEngine(market_data=service, risk_free_rate=0.02)


def test_default_methods_run_end_to_end(engine):
    assert engine.available_methods() == ["rule_based", "mean_variance", "research_informed"]
    rule = engine.optimize("rule_based", PROFILE)
    mvo = engine.optimize("mean_variance", PROFILE)
    research = engine.optimize("research_informed", PROFILE)
    for result in (rule, mvo, research):
        assert (result.weights >= 0).all() and result.weights.sum() == pytest.approx(1.0)
    assert set(mvo.weights.index) == set(get_tickers())
    assert mvo.frontier is not None and len(mvo.frontier) > 10
    assert rule.details["estimation_window"]["observations"] >= 24


def test_inputs_are_memoized_and_clearable(engine):
    first = engine.market_inputs()
    assert engine.market_inputs() is first
    calls = sum(engine.market_data.connector.history_calls.values())
    engine.clear_inputs_cache()
    assert engine.market_inputs() is not first
    assert sum(engine.market_data.connector.history_calls.values()) == calls  # served from disk cache


def test_strategy_options_and_ticker_subset(engine):
    result = engine.optimize("mean_variance", PROFILE, tickers=["SPY", "AGG", "VNQ"], objective="max_sharpe", n_points=10)
    assert list(result.weights.index) == ["SPY", "AGG", "VNQ"]
    assert result.details["objective"] == "max_sharpe"


def test_compare_and_custom_strategy(engine):
    engine.register(EqualWeight())
    results = engine.compare(PROFILE)
    assert list(results) == ["rule_based", "mean_variance", "research_informed", "equal_weight"]
    assert results["equal_weight"].weights.nunique() == 1
    with pytest.raises(ValueError):
        engine.register(EqualWeight())
    engine.register(EqualWeight(), replace=True)


def test_unknown_method(engine):
    with pytest.raises(KeyError, match="available"):
        engine.optimize("black_litterman", PROFILE)


def test_insufficient_history_raises(tmp_path):
    service = MarketDataService(FakeConnector(), ParquetCache(tmp_path / "c"))  # ~14 months of data
    engine = PortfolioOptimizationEngine(market_data=service)
    with pytest.raises(OptimizationError, match="observations"):
        engine.optimize("mean_variance", PROFILE)


def test_default_cash_return_zero_and_sharpe_risk_free_rate_four_percent(tmp_path):
    service = MarketDataService(LongHistoryConnector(), ParquetCache(tmp_path / "rf"))
    engine = PortfolioOptimizationEngine(market_data=service)
    inputs = engine.market_inputs()
    assert engine.risk_free_rate == inputs.risk_free_rate == 0.04
    assert engine.cash_return == inputs.cash_return == 0.0
    other = PortfolioOptimizationEngine(market_data=service, risk_free_rate=0.03, cash_return=0.01)
    assert (other.market_inputs().risk_free_rate, other.market_inputs().cash_return) == (0.03, 0.01)
