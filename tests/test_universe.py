import pytest

from portfolio_builder import universe as u


def test_tickers_in_order():
    assert u.get_tickers() == ["SPY", "VTI", "EFA", "VXUS", "EEM", "VWO", "AGG", "BND", "TIP", "VTIP", "VNQ"]


def test_asset_classes_names_and_roles():
    assert [(ac.name, ac.role, ac.tickers) for ac in u.get_asset_classes()] == [
        ("US large-cap stocks", "Growth, domestic equity exposure", ("SPY", "VTI")),
        ("International developed stocks", "Diversification, international exposure", ("EFA", "VXUS")),
        ("Emerging market stocks", "Higher growth potential, higher risk", ("EEM", "VWO")),
        ("US Aggregate bonds", "Stability, income", ("AGG", "BND")),
        ("Treasury inflation-protected securities", "Inflation Hedge", ("TIP", "VTIP")),
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
