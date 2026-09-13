import numpy as np
import pandas as pd
import pytest

from portfolio_builder.optimization import CASH, InvestorProfile, MarketInputs, RuleBasedConfig, RuleBasedStrategy
from portfolio_builder.optimization import rule_based as rb
from portfolio_builder.universe import AssetClass, get_asset_class, get_tickers

VOLS = {"VTI": 0.16, "VXUS": 0.165, "VWO": 0.20, "BND": 0.052, "VTIP": 0.025, "VNQ": 0.20,
        "SPY": 0.15, "TIP": 0.06}  # SPY/TIP only used by the multi-fund blending test


def universe_inputs(tickers: list[str] | None = None) -> MarketInputs:
    tickers = tickers or get_tickers()
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


def test_each_asset_class_holds_its_single_fund():
    inputs = universe_inputs()
    result = RuleBasedStrategy().allocate(InvestorProfile(40, 5.5), inputs)
    classes = result.details["asset_class_weights"]
    fund_for = {"us_large_cap": "VTI", "intl_developed": "VXUS", "emerging_markets": "VWO",
                "us_aggregate_bonds": "BND", "tips": "VTIP", "real_estate": "VNQ"}
    for key, fund in fund_for.items():
        assert result.weights[fund] == pytest.approx(classes[key])
    assert set(result.weights.index) == set(get_tickers()) | {CASH}
    assert result.details["ticker_order_by_volatility"]["tips"] == ["VTIP"]


def test_multi_fund_class_blends_by_risk_tolerance(monkeypatch):
    """The blending rule still works if an asset class is given more than one fund."""
    two_funds = {"us_large_cap": ("SPY", "VTI"), "tips": ("TIP", "VTIP")}

    def patched(key):
        ac = get_asset_class(key)
        return AssetClass(ac.key, ac.name, ac.role, two_funds.get(key, ac.tickers))

    monkeypatch.setattr(rb, "get_asset_class", patched)
    inputs = universe_inputs([*get_tickers(), "SPY", "TIP"])
    conservative = RuleBasedStrategy().allocate(InvestorProfile(40, 1), inputs).weights
    aggressive = RuleBasedStrategy().allocate(InvestorProfile(40, 10), inputs).weights
    middle = RuleBasedStrategy().allocate(InvestorProfile(40, 5.5), inputs).weights
    for calm, volatile in [("SPY", "VTI"), ("VTIP", "TIP")]:
        assert conservative.get(volatile, 0.0) == 0 and conservative[calm] > 0
        assert aggressive.get(calm, 0.0) == 0 and aggressive[volatile] > 0
        assert middle[calm] == pytest.approx(middle[volatile])


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
