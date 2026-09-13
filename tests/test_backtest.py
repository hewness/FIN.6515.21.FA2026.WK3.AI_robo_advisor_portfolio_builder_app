import numpy as np
import pandas as pd
import pytest

from portfolio_builder.backtest import BacktestConfig, BacktestError, Backtester
from portfolio_builder.market_data import MarketDataService, ParquetCache
from portfolio_builder.optimization import CASH

from .conftest import FakeConnector, make_history

INDEX = pd.bdate_range("2012-01-02", periods=2600, name="date")  # ~10 years


def history_from_prices(prices: pd.Series) -> pd.DataFrame:
    frame = make_history(start=str(prices.index[0].date()), periods=len(prices))
    frame.index = prices.index
    for col in ("open", "high", "low", "close", "adj_close"):
        frame[col] = prices.to_numpy()
    return frame


class PathConnector(FakeConnector):
    """Serves explicit price paths; any other ticker gets a flat price on INDEX."""

    def __init__(self, paths: dict[str, pd.Series]):
        super().__init__()
        self.paths = paths

    def fetch_history(self, ticker, start=None, end=None):
        self.history_calls[ticker] += 1
        prices = self.paths.get(ticker, pd.Series(100.0, index=INDEX))
        return history_from_prices(prices)


def backtester(tmp_path, paths) -> Backtester:
    return Backtester(MarketDataService(PathConnector(paths), ParquetCache(tmp_path / "cache")))


def growth(rate: float, index=INDEX, start: float = 100.0) -> pd.Series:
    return pd.Series(start * (1 + rate) ** np.arange(len(index)), index=index)


def reference_values(prices: pd.DataFrame, weights: pd.Series, start, code: str, initial: float) -> pd.Series:
    """Independent daily loop: drift holdings, rebalance on the first day of each new period."""
    rets = prices.pct_change().loc[prices.index > start]
    value, holdings, prev = initial, initial * weights.to_numpy(), None
    out = {start: initial}
    for date, row in rets.iterrows():
        period = date.to_period(code)
        if prev is not None and period != prev:
            holdings = value * weights.to_numpy()
        holdings = holdings * (1 + row[weights.index].to_numpy())
        value = holdings.sum()
        out[date] = value
        prev = period
    return pd.Series(out)


def test_constant_growth_matches_closed_form(tmp_path):
    bt = backtester(tmp_path, {"SPY": growth(0.0004)})
    result = bt.run({"p": pd.Series({"SPY": 1.0})}, BacktestConfig(years=5), initial_value=10_000)
    days = len(result.values) - 1
    expected_final = 10_000 * 1.0004**days
    m = result.metrics["p"]
    assert m.final_value == pytest.approx(expected_final)
    assert m.cagr == pytest.approx((expected_final / 10_000) ** (1 / result.years_covered) - 1, rel=1e-9)
    assert m.max_drawdown == 0 and m.max_drawdown_date is None
    pd.testing.assert_series_equal(result.values["p"], result.values["benchmark"], check_names=False)
    assert result.years_covered == pytest.approx(5, abs=0.02)


def test_max_drawdown_of_known_path(tmp_path):
    n = len(INDEX)
    path = np.concatenate([np.linspace(100, 120, n // 3), np.linspace(120, 60, n // 3),
                           np.linspace(60, 150, n - 2 * (n // 3))])
    prices = pd.Series(path, index=INDEX)
    bt = backtester(tmp_path, {"SPY": prices})
    result = bt.run({"p": pd.Series({"SPY": 1.0})}, BacktestConfig(years=9), initial_value=1_000)
    m = result.metrics["p"]
    assert m.max_drawdown == pytest.approx(-0.5, abs=1e-9)
    assert m.max_drawdown_date == prices.idxmin().date().isoformat()
    assert result.drawdowns["p"].max() == 0


@pytest.mark.parametrize("rebalance,code", [("monthly", "M"), ("quarterly", "Q"), ("annual", "Y")])
def test_rebalancing_matches_reference(tmp_path, rebalance, code):
    rng = np.random.default_rng(3)
    spy = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.012, len(INDEX))), index=INDEX)
    agg = pd.Series(100 * np.cumprod(1 + rng.normal(0.0001, 0.003, len(INDEX))), index=INDEX)
    bt = backtester(tmp_path, {"SPY": spy, "AGG": agg})
    weights = pd.Series({"SPY": 0.6, "AGG": 0.4})
    result = bt.run({"p": weights}, BacktestConfig(years=8, rebalance=rebalance), initial_value=5_000)
    ref = reference_values(pd.DataFrame({"SPY": spy, "AGG": agg}), weights, result.start, code, 5_000)
    np.testing.assert_allclose(result.values["p"].to_numpy(), ref.to_numpy(), rtol=1e-10)


def test_rebalancing_frequency_changes_result(tmp_path):
    rng = np.random.default_rng(5)
    spy = pd.Series(100 * np.cumprod(1 + rng.normal(0.0006, 0.015, len(INDEX))), index=INDEX)
    bt = backtester(tmp_path, {"SPY": spy})
    weights = {"p": pd.Series({"SPY": 0.5, "AGG": 0.5})}
    monthly = bt.run(weights, BacktestConfig(years=8, rebalance="monthly")).metrics["p"].final_value
    annual = bt.run(weights, BacktestConfig(years=8, rebalance="annual")).metrics["p"].final_value
    assert monthly != pytest.approx(annual, rel=1e-6)


def test_proxy_fills_before_inception(tmp_path):
    late = INDEX[INDEX >= "2016-01-04"]
    tip = growth(0.0002)
    vtip = growth(0.0001, index=late, start=50.0)
    bt = backtester(tmp_path, {"TIP": tip, "VTIP": vtip})
    assert bt.proxy_map() == {"VTI": "SPY", "VXUS": "EFA", "VWO": "EEM", "BND": "AGG", "VTIP": "TIP"}
    assert bt.proxy_map(["VTIP", "VNQ"]) == {"VTIP": "TIP"}

    result = bt.run({"p": pd.Series({"VTIP": 1.0})}, BacktestConfig(years=9), initial_value=100)
    assert result.proxies_used["VTIP"]["proxy"] == "TIP"
    assert result.proxies_used["VTIP"]["until"] == "2016-01-05"  # first VTIP return day
    before = result.values["p"].loc[: "2016-01-04"]
    np.testing.assert_allclose(before.pct_change().dropna(), 0.0002, rtol=1e-9)
    after = result.values["p"].loc["2016-01-06":]
    np.testing.assert_allclose(after.pct_change().dropna(), 0.0001, rtol=1e-9)
    assert any("older ETF tracking the same asset class" in n and "never held" in n for n in result.notes)

    no_proxy = bt.run({"p": pd.Series({"VTIP": 1.0})}, BacktestConfig(years=9, use_proxies=False))
    assert no_proxy.start >= pd.Timestamp("2016-01-04")
    assert any("Only" in n for n in no_proxy.notes)


def test_short_history_clamps_window(tmp_path):
    bt = backtester(tmp_path, {"SPY": growth(0.0003)})
    result = bt.run({"p": pd.Series({"SPY": 1.0})}, BacktestConfig(years=20))
    assert result.start == INDEX[0]
    assert result.years_requested == 20 and result.years_covered < 11
    assert any("Only" in n and "years of history" in n for n in result.notes)


def test_cash_accrues_at_risk_free_rate(tmp_path):
    bt = backtester(tmp_path, {})
    result = bt.run({"cash": pd.Series({CASH: 1.0})}, BacktestConfig(years=3, risk_free_rate=0.05), initial_value=1_000)
    days = len(result.values) - 1
    assert result.metrics["cash"].final_value == pytest.approx(1_000 * 1.05 ** (days / 252))
    assert result.metrics["cash"].volatility == pytest.approx(0.0, abs=1e-12)


def test_best_and_worst_years_exclude_partial_final_year(tmp_path):
    bt = backtester(tmp_path, {"SPY": growth(0.0004)})
    m = bt.run({"p": pd.Series({"SPY": 1.0})}, BacktestConfig(years=8)).metrics["p"]
    assert m.best_year == pytest.approx(m.worst_year, rel=0.02)  # constant growth: every full year alike
    assert m.best_year == pytest.approx(1.0004**261 - 1, rel=0.01)


def test_invalid_inputs(tmp_path):
    bt = backtester(tmp_path, {})
    with pytest.raises(BacktestError):
        bt.run({}, BacktestConfig())
    with pytest.raises(BacktestError):
        bt.run({"benchmark": pd.Series({"SPY": 1.0})})
    with pytest.raises(BacktestError):
        bt.run({"p": pd.Series({"SPY": 0.5})})
    with pytest.raises(ValueError):
        BacktestConfig(rebalance="weekly")
