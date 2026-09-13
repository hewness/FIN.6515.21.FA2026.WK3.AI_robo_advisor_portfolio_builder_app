import pytest

from portfolio_builder import universe as u


def test_tickers_in_order():
    assert u.get_tickers() == ["VTI", "VXUS", "VWO", "BND", "VTIP", "VNQ"]


def test_asset_classes_names_and_roles():
    assert [(ac.name, ac.role, ac.tickers) for ac in u.get_asset_classes()] == [
        ("US large-cap stocks", "Growth, domestic equity exposure", ("VTI",)),
        ("International developed stocks", "Diversification, international exposure", ("VXUS",)),
        ("Emerging market stocks", "Higher growth potential, higher risk", ("VWO",)),
        ("US Aggregate bonds", "Stability, income", ("BND",)),
        ("Treasury inflation-protected securities", "Inflation Hedge", ("VTIP",)),
        ("Real estate (REITs)", "Real asset diversification", ("VNQ",)),
        ("Cash and Money Markets", "Liquidity, capital preservation", ()),
    ]


def test_cash_has_no_tickers():
    assert u.get_asset_class("cash").tickers == ()
    assert "cash" not in u.tickers_by_asset_class()
    assert u.tickers_by_asset_class(include_empty=True)["cash"] == []


def test_ticker_lookup():
    assert u.get_asset_class_for_ticker("vwo").key == "emerging_markets"
    with pytest.raises(KeyError):
        u.get_asset_class_for_ticker("QQQ")


def test_universe_frame():
    frame = u.universe_frame()
    assert list(frame.index) == u.get_tickers()
    assert frame.loc["VNQ", "role"] == "Real asset diversification"
    assert frame.index.is_unique


def test_one_ticker_per_asset_class():
    for ac in u.get_asset_classes():
        assert len(ac.tickers) == (0 if ac.key == "cash" else 1)


def test_history_proxies_are_backtest_only():
    proxies = u.get_history_proxies()
    assert proxies == {"VTI": "SPY", "VXUS": "EFA", "VWO": "EEM", "BND": "AGG", "VTIP": "TIP"}
    assert set(proxies) <= set(u.get_tickers())
    assert not set(proxies.values()) & set(u.get_tickers())  # never investable
    proxies["VTI"] = "X"
    assert u.get_history_proxies()["VTI"] == "SPY"  # returns a copy
