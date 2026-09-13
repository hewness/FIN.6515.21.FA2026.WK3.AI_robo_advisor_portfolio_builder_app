import numpy as np
import pandas as pd
import pytest

from portfolio_builder.market_data import MarketDataService, get_market_data_service
from portfolio_builder.universe import get_tickers

from .conftest import FakeConnector


def test_cache_miss_downloads_then_hits(service, connector, cache):
    first = service.get_history("SPY")
    assert connector.history_calls["SPY"] == 1
    assert cache.has_history("SPY")

    second = service.get_history("spy")
    assert connector.history_calls["SPY"] == 1  # served from disk
    pd.testing.assert_frame_equal(first, second)


def test_refresh_flag_redownloads(service, connector):
    old = service.get_history("SPY")
    connector.generation = 1
    new = service.get_history("SPY", refresh=True)
    assert connector.history_calls["SPY"] == 2
    assert not new["adj_close"].equals(old["adj_close"])


def test_refresh_all_and_failure_keeps_old_data(cache):
    good = FakeConnector()
    svc = MarketDataService(good, cache, default_tickers=["SPY", "BND"])
    svc.ensure_cached()
    before = cache.read_history("BND")
    before_info = cache.read_info("BND")

    failing = FakeConnector(fail={"BND"})
    failing.generation = 1
    results = MarketDataService(failing, cache, default_tickers=["SPY", "BND"]).refresh()

    assert results["SPY"] == "refreshed"
    assert results["BND"].startswith("failed")
    pd.testing.assert_frame_equal(cache.read_history("BND"), before)
    assert cache.read_info("BND") == before_info


def test_ensure_cached_only_downloads_missing(service, connector):
    service.get_history("SPY")
    service.get_info("SPY")
    results = service.ensure_cached(["SPY", "VTI"])
    assert results == {"SPY": "cached", "VTI": "downloaded"}
    assert connector.history_calls["SPY"] == 1
    assert connector.history_calls["VTI"] == 1


def test_default_tickers_are_universe(service):
    prices = service.get_prices()
    assert list(prices.columns) == get_tickers()


def test_get_prices_alignment(cache):
    connector = FakeConnector(starts={"VTIP": "2020-06-01"})
    svc = MarketDataService(connector, cache)

    inner = svc.get_prices(["SPY", "VTIP"])
    assert inner.index.min() == pd.Timestamp("2020-06-01")
    assert not inner.isna().any().any()

    outer = svc.get_prices(["SPY", "VTIP"], align="outer")
    assert outer.index.min() == pd.Timestamp("2020-01-01")
    assert outer.loc["2020-01-01", "VTIP"] != outer.loc["2020-01-01", "VTIP"]  # NaN

    window = svc.get_prices(["SPY"], start="2020-03-02", end="2020-03-06")
    assert len(window) == 5


def test_get_prices_rejects_bad_args(service):
    with pytest.raises(ValueError):
        service.get_prices(["SPY"], field="price")
    with pytest.raises(ValueError):
        service.get_prices([])


def test_returns_daily_and_log(service):
    prices = service.get_prices(["SPY", "VTI"])
    simple = service.get_returns(["SPY", "VTI"])
    assert len(simple) == len(prices) - 1
    np.testing.assert_allclose(simple["SPY"].iloc[0], prices["SPY"].iloc[1] / prices["SPY"].iloc[0] - 1)

    log = service.get_returns(["SPY", "VTI"], method="log")
    np.testing.assert_allclose(np.exp(log) - 1, simple)


def test_returns_monthly(service):
    prices = service.get_prices(["SPY"])  # 300 bdays from 2020-01-01: ends mid-Feb 2021
    monthly = service.get_returns(["SPY"], frequency="monthly")
    month_end = prices.resample("ME").last()
    assert prices.index.max() < pd.Timestamp("2021-02-26")
    assert len(monthly) == len(month_end) - 2  # first month has no prior; partial Feb 2021 dropped
    assert monthly.index.max() == pd.Timestamp("2021-01-31")
    assert service.get_returns(["SPY"], frequency="monthly", include_partial=True).index.max() == pd.Timestamp(
        "2021-02-28"
    )
    np.testing.assert_allclose(monthly["SPY"].iloc[0], month_end["SPY"].iloc[1] / month_end["SPY"].iloc[0] - 1)
    assert service.get_returns(["SPY"], frequency="weekly").index[0].day_name() == "Friday"


def test_dividends_and_metadata(service):
    divs = service.get_dividends(["SPY", "AGG"])
    assert (divs == 0).all().all()

    meta = service.get_universe_metadata()
    assert list(meta.index) == get_tickers()
    assert {"asset_class", "role", "longName", "netExpenseRatio"} <= set(meta.columns)
    assert meta.loc["VNQ", "asset_class"] == "Real estate (REITs)"


def test_cache_status(service):
    service.get_history("VTI")
    status = service.cache_status()
    assert list(status.index) == get_tickers()
    assert bool(status.loc["VTI", "cached"]) and not bool(status.loc["VXUS", "cached"])
    assert status.loc["VTI", "asset_class"] == "US large-cap stocks"


def test_factory_uses_env_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(tmp_path / "envcache"))
    svc = get_market_data_service(connector=FakeConnector())
    assert svc.cache.root == tmp_path / "envcache"
