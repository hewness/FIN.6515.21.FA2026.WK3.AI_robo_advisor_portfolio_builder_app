import pandas as pd
import pytest

from portfolio_builder.optimization import (
    CASH,
    InvestorProfile,
    OptimizationError,
    PracticalFinanceAssumptions,
    ResearchInformedConfig,
    ResearchInformedStrategy,
    human_capital,
    merton_share,
    research_equity_share,
    risk_aversion,
)
from portfolio_builder.optimization.inputs import MarketInputs
from portfolio_builder.optimization.research_informed import (
    human_capital_schedule,
    retirement_discount_rate,
    working_discount_rate,
)

# Practical Finance's worked example (Section 3.3, Table 3): 55-year-old, relative risk aversion 7, 2% log real
# risk-free rate, 2% log equity premium, 40% replacement rate, college-graduate income risk, $100k income through
# age 66 and $40k afterwards, $1,000,000 financial portfolio.
PAPER = PracticalFinanceAssumptions(log_equity_premium=0.02)


def test_merton_share_matches_paper():
    assert merton_share(7, PAPER) == pytest.approx(0.155, abs=5e-4)
    assert merton_share(0.1) == 1.0  # clipped


def test_discount_rates_match_paper():
    assert working_discount_rate(55, 7, 0.40, PAPER) == pytest.approx(0.0981, abs=5e-5)
    assert working_discount_rate(56, 7, 0.40, PAPER) == pytest.approx(0.0981, abs=5e-5)
    assert retirement_discount_rate(66, 7, PAPER) == pytest.approx(0.0334, abs=5e-5)


def test_human_capital_matches_table_3():
    schedule = human_capital_schedule(55, 100_000, 40_000, 67, 7, PAPER)
    assert list(schedule.index[[0, -1]]) == [56, 100]
    assert schedule.loc[67, "gross_discount_rate"] == pytest.approx(1.0334, abs=5e-5)
    assert schedule.loc[66, "cumulative_discount_rate"] == pytest.approx(2.8222, abs=5e-4)
    assert schedule.loc[100, "discounted_income"] == pytest.approx(4_148, abs=1)
    assert human_capital(55, 100_000, 40_000, 67, 7, PAPER) == pytest.approx(924_805, abs=1)


def test_equity_share_matches_paper_example():
    profile = InvestorProfile(age=55, risk_tolerance=4, annual_income=100_000, retirement_income=40_000,
                              financial_wealth=1_000_000)
    assumptions = PracticalFinanceAssumptions(log_equity_premium=0.02, gamma_bounds=(7.0, 7.0))
    result = research_equity_share(profile, assumptions)
    assert result.risk_aversion == 7
    assert result.equity == pytest.approx(0.30, abs=0.005)
    assert result.human_capital_ratio == pytest.approx(0.9248, abs=1e-4)


def test_risk_aversion_mapping():
    assert risk_aversion(1) == 10 and risk_aversion(10) == 4
    assert risk_aversion(5.5) == pytest.approx(7.0) and risk_aversion(6.5) == pytest.approx(6 + 1 / 3)


def _share(**kwargs):
    base = dict(age=68, risk_tolerance=6.5, annual_income=0, retirement_income=30_000, financial_wealth=500_000)
    return research_equity_share(InvestorProfile(**{**base, **kwargs}))


def test_retiree_lands_in_duarte_range():
    result = _share()
    assert 0.55 <= result.equity <= 0.65
    assert result.human_capital == pytest.approx(583_860, rel=1e-3)


def test_wealth_income_and_age_effects():
    assert _share(financial_wealth=1_000_000).equity < _share().equity < _share(financial_wealth=250_000).equity
    assert _share(retirement_income=40_000).equity > _share().equity
    assert _share(age=80).equity < _share().equity  # fewer years of benefits left
    no_income = _share(retirement_income=0)
    assert no_income.human_capital == 0 and no_income.equity == pytest.approx(no_income.merton_share)
    young = _share(age=30, annual_income=80_000, retirement_income=None, financial_wealth=50_000)
    assert young.equity == 1.0 and young.unclipped_equity > 1  # human capital dwarfs savings
    assert _share(risk_tolerance=2).equity < _share(risk_tolerance=9).equity


def test_default_retirement_income_is_40_percent_of_income():
    profile = InvestorProfile(age=40, risk_tolerance=5, annual_income=90_000)
    assert profile.resolved_retirement_income == pytest.approx(36_000)
    with pytest.raises(ValueError):
        InvestorProfile(age=40, risk_tolerance=5, financial_wealth=0)
    with pytest.raises(ValueError):
        InvestorProfile(age=40, risk_tolerance=5, annual_income=-1)


def _inputs() -> MarketInputs:
    tickers = ["VTI", "VXUS", "VWO", "VNQ", "BND", "VTIP"]
    mu = pd.Series([0.09, 0.07, 0.08, 0.075, 0.03, 0.025], index=tickers)
    vols = pd.Series([0.16, 0.17, 0.2, 0.2, 0.05, 0.03], index=tickers)
    cov = pd.DataFrame(0.0, index=tickers, columns=tickers)
    for t in tickers:
        cov.loc[t, t] = vols[t] ** 2
    return MarketInputs(expected_returns=mu, covariance=cov, risk_free_rate=0.0, observations=60)


def test_strategy_allocates_by_sleeves():
    profile = InvestorProfile(age=68, risk_tolerance=6.5, retirement_income=30_000, financial_wealth=500_000)
    result = ResearchInformedStrategy().allocate(profile, _inputs())
    w, equity = result.weights, result.details["equity_pct"]
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    assert w[["VTI", "VXUS", "VWO", "VNQ"]].sum() == pytest.approx(equity)
    cfg = ResearchInformedConfig()
    assert w["VXUS"] == pytest.approx(equity * cfg.equity_sleeve["intl_developed"])
    assert w[CASH] == pytest.approx((1 - equity) * cfg.defensive_sleeve["cash"])
    assert result.details["human_capital"] > 0 and result.method == "research_informed"


def test_strategy_requires_financial_wealth():
    with pytest.raises(OptimizationError, match="financial wealth"):
        ResearchInformedStrategy().allocate(InvestorProfile(age=40, risk_tolerance=5), _inputs())


def test_config_rejects_bad_sleeves():
    with pytest.raises(ValueError):
        ResearchInformedConfig(equity_sleeve={"us_large_cap": 0.5})
