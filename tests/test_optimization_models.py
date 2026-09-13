import numpy as np
import pandas as pd
import pytest

from portfolio_builder.optimization import (
    CASH,
    AllocationResult,
    InvestorProfile,
    MarketInputs,
    OptimizationError,
    PortfolioMetrics,
    portfolio_metrics,
)

METRICS = PortfolioMetrics(0.05, 0.1, 0.1)


def make_inputs(rf: float = 0.04) -> MarketInputs:
    tickers = ["SPY", "BND"]
    return MarketInputs(
        expected_returns=pd.Series([0.10, 0.03], index=tickers),
        covariance=pd.DataFrame([[0.04, 0.002], [0.002, 0.0025]], index=tickers, columns=tickers),
        risk_free_rate=rf,
    )


@pytest.mark.parametrize("age,risk", [(17, 5), (101, 5), (40, 0.5), (40, 10.5)])
def test_profile_validation(age, risk):
    with pytest.raises(ValueError):
        InvestorProfile(age=age, risk_tolerance=risk)


def test_risk_fraction():
    assert InvestorProfile(40, 1).risk_fraction == 0.0
    assert InvestorProfile(40, 10).risk_fraction == 1.0
    assert InvestorProfile(40, 5.5).risk_fraction == pytest.approx(0.5)


@pytest.mark.parametrize(
    "weights",
    [{"SPY": 1.1, "BND": -0.1}, {"SPY": 0.5, "BND": 0.4}, {"SPY": float("nan"), "BND": 1.0}, {}],
)
def test_result_rejects_invalid_weights(weights):
    with pytest.raises(OptimizationError):
        AllocationResult("x", InvestorProfile(40, 5), pd.Series(weights, dtype=float), METRICS)


def test_result_cleans_float_noise():
    result = AllocationResult("x", InvestorProfile(40, 5), pd.Series({"SPY": 1.0 + 5e-7, "BND": -5e-7}), METRICS)
    assert (result.weights >= 0).all()
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_asset_class_weights_include_cash():
    weights = pd.Series({"SPY": 0.3, "VTI": 0.2, "BND": 0.4, CASH: 0.1})
    result = AllocationResult("x", InvestorProfile(40, 5), weights, METRICS)
    classes = result.asset_class_weights
    assert classes["US large-cap stocks"] == pytest.approx(0.5)
    assert classes["Cash and Money Markets"] == pytest.approx(0.1)
    frame = result.to_frame()
    assert frame.loc[CASH, "role"] == "Liquidity, capital preservation"
    assert list(frame["weight"]) == sorted(frame["weight"], reverse=True)


def test_portfolio_metrics_with_cash():
    inputs = make_inputs(rf=0.04)
    m = portfolio_metrics(pd.Series({"SPY": 0.5, CASH: 0.5}), inputs)
    assert m.expected_return == pytest.approx(0.5 * 0.10 + 0.5 * 0.04)
    assert m.volatility == pytest.approx(0.5 * 0.2)
    assert m.sharpe_ratio == pytest.approx((0.07 - 0.04) / 0.1)

    all_cash = portfolio_metrics(pd.Series({CASH: 1.0}), inputs)
    assert all_cash.volatility == 0 and all_cash.sharpe_ratio == 0


def test_portfolio_metrics_rejects_unknown_ticker():
    with pytest.raises(OptimizationError):
        portfolio_metrics(pd.Series({"QQQ": 1.0}), make_inputs())


def test_inputs_from_returns_annualizes():
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0.01, 0.03, (36, 2)), index=idx, columns=["SPY", "BND"])
    inputs = MarketInputs.from_returns(returns, "monthly", 0.02)
    np.testing.assert_allclose(inputs.expected_returns, returns.mean() * 12)
    np.testing.assert_allclose(inputs.covariance, returns.cov() * 12)
    assert inputs.observations == 36 and inputs.estimation_window["start"] == "2020-01-31"
    np.testing.assert_allclose(np.diag(inputs.correlation), 1.0)

    with pytest.raises(OptimizationError):
        MarketInputs.from_returns(returns.iloc[:10])
