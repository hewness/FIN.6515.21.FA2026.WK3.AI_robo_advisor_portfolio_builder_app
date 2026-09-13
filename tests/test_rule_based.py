import numpy as np
import pandas as pd
import pytest

from portfolio_builder.optimization import CASH, InvestorProfile, MarketInputs, RuleBasedConfig, RuleBasedStrategy
from portfolio_builder.universe import get_tickers

# Per-ticker volatility; within each pair the second listed ticker is the more volatile one.
VOLS = {
    "SPY": 0.15, "VTI": 0.16, "EFA": 0.17, "VXUS": 0.165, "EEM": 0.21, "VWO": 0.20,
    "AGG": 0.050, "BND": 0.052, "TIP": 0.06, "VTIP": 0.025, "VNQ": 0.20,
}


def universe_inputs() -> MarketInputs:
    tickers = get_tickers()
    vol = np.array([VOLS[t] for t in tickers])
    cov = np.diag(vol**2)
    mu = pd.Series(np.linspace(0.02, 0.09, len(tickers)), index=tickers)
    return MarketInputs(mu, pd.DataFrame(cov, index=tickers, columns=tickers), 0.04)


@pytest.mark.parametrize("age,risk,expected", [(40, 1, 0.50), (40, 10, 0.90), (40, 5.5, 0.70), (25, 10, 1.0), (95, 1, 0.10)])
def test_equity_fraction(age, risk, expected):
    equity, _ = RuleBasedStrategy().equity_fraction(InvestorProfile(age, risk))
    assert equity == pytest.approx(expected)


def test_sleeves_and_cash():
    result = RuleBasedStrategy().allocate(InvestorProfile(40, 1), universe_inputs())
    classes = result.details["asset_class_weights"]
    assert classes["us_large_cap"] == pytest.approx(0.5 * 0.55)
    assert classes["us_aggregate_bonds"] == pytest.approx(0.5 * 0.70)
    assert result.weights[CASH] == pytest.approx(0.5 * 0.10)
    assert result.asset_class_weights["Cash and Money Markets"] == pytest.approx(0.05)
    assert (result.weights >= 0).all() and result.weights.sum() == pytest.approx(1.0)


def test_ticker_selection_by_risk_tolerance():
    inputs = universe_inputs()
    conservative = RuleBasedStrategy().allocate(InvestorProfile(40, 1), inputs).weights
    aggressive = RuleBasedStrategy().allocate(InvestorProfile(40, 10), inputs).weights
    middle = RuleBasedStrategy().allocate(InvestorProfile(40, 5.5), inputs).weights

    for calm, volatile in [("SPY", "VTI"), ("VXUS", "EFA"), ("VWO", "EEM"), ("AGG", "BND"), ("VTIP", "TIP")]:
        assert conservative.get(volatile, 0.0) == 0 and conservative[calm] > 0
        assert aggressive.get(calm, 0.0) == 0 and aggressive[volatile] > 0
        assert middle[calm] == pytest.approx(middle[volatile])
    assert conservative["VNQ"] > 0 and aggressive["VNQ"] > 0
    order = RuleBasedStrategy().allocate(InvestorProfile(40, 1), inputs).details["ticker_order_by_volatility"]
    assert order["tips"] == ["VTIP", "TIP"]


def test_constraints_hold_across_profiles():
    inputs = universe_inputs()
    for age in (18, 45, 70, 100):
        for risk in range(1, 11):
            w = RuleBasedStrategy().allocate(InvestorProfile(age, risk), inputs).weights
            assert (w >= 0).all()
            assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_metrics_use_market_inputs():
    inputs = universe_inputs()
    result = RuleBasedStrategy().allocate(InvestorProfile(30, 7), inputs)
    risky = result.weights.drop(CASH)
    expected = float(risky @ inputs.expected_returns[risky.index]) + result.weights[CASH] * 0.04
    assert result.metrics.expected_return == pytest.approx(expected)


def test_custom_config_validation():
    with pytest.raises(ValueError):
        RuleBasedConfig(equity_sleeve={"us_large_cap": 0.5})
    with pytest.raises(KeyError):
        RuleBasedConfig(defensive_sleeve={"crypto": 1.0})
    cfg = RuleBasedConfig(base=120, defensive_sleeve={"us_aggregate_bonds": 1.0})
    result = RuleBasedStrategy(cfg).allocate(InvestorProfile(60, 5.5), universe_inputs())
    assert result.details["equity_pct"] == pytest.approx(0.60)
    assert CASH not in result.weights.index
