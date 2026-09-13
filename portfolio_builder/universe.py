"""Investment universe: asset classes, their portfolio roles, and ticker mapping."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AssetClass:
    key: str
    name: str
    role: str
    tickers: tuple[str, ...] = ()


ASSET_CLASSES: tuple[AssetClass, ...] = (
    AssetClass("us_large_cap", "US large-cap stocks", "Growth, domestic equity exposure", ("VTI",)),
    AssetClass("intl_developed", "International developed stocks", "Diversification, international exposure", ("VXUS",)),
    AssetClass("emerging_markets", "Emerging market stocks", "Higher growth potential, higher risk", ("VWO",)),
    AssetClass("us_aggregate_bonds", "US Aggregate bonds", "Stability, income", ("BND",)),
    AssetClass("tips", "Treasury inflation-protected securities", "Inflation Hedge", ("VTIP",)),
    AssetClass("real_estate", "Real estate (REITs)", "Real asset diversification", ("VNQ",)),
    # Cash has no ticker; it is modeled outside the market data feed.
    AssetClass("cash", "Cash and Money Markets", "Liquidity, capital preservation"),
)

# Backtest-only history proxies: older ETFs tracking the same asset class whose returns stand in before a
# universe fund's inception, so 10-20 year backtests remain possible. They are never recommended or held.
HISTORY_PROXIES: dict[str, str] = {"VTI": "SPY", "VXUS": "EFA", "VWO": "EEM", "BND": "AGG", "VTIP": "TIP"}

_BY_KEY: dict[str, AssetClass] = {ac.key: ac for ac in ASSET_CLASSES}
_BY_TICKER: dict[str, AssetClass] = {}
for _ac in ASSET_CLASSES:
    for _ticker in _ac.tickers:
        if _ticker in _BY_TICKER:
            raise ValueError(f"Ticker {_ticker} is assigned to more than one asset class")
        _BY_TICKER[_ticker] = _ac
for _fund, _proxy in HISTORY_PROXIES.items():
    if _fund not in _BY_TICKER or _proxy in _BY_TICKER:
        raise ValueError(f"History proxy {_proxy} for {_fund} must map a universe fund to a non-universe ticker")


def get_asset_classes() -> tuple[AssetClass, ...]:
    return ASSET_CLASSES


def get_history_proxies() -> dict[str, str]:
    """Universe fund -> older same-asset-class ETF used only to extend backtest history."""
    return dict(HISTORY_PROXIES)


def get_tickers() -> list[str]:
    """All tradable tickers in the universe, in asset-class order."""
    return [t for ac in ASSET_CLASSES for t in ac.tickers]


def get_asset_class(key: str) -> AssetClass:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"Unknown asset class: {key!r}") from None


def get_asset_class_for_ticker(ticker: str) -> AssetClass:
    try:
        return _BY_TICKER[ticker.upper()]
    except KeyError:
        raise KeyError(f"Ticker not in universe: {ticker!r}") from None


def tickers_by_asset_class(include_empty: bool = False) -> dict[str, list[str]]:
    """Map of asset class key -> tickers, e.g. for group weight constraints."""
    return {ac.key: list(ac.tickers) for ac in ASSET_CLASSES if ac.tickers or include_empty}


def universe_frame() -> pd.DataFrame:
    """One row per ticker with its asset class and portfolio role, indexed by ticker."""
    rows = [
        {"ticker": t, "asset_class_key": ac.key, "asset_class": ac.name, "role": ac.role}
        for ac in ASSET_CLASSES
        for t in ac.tickers
    ]
    return pd.DataFrame(rows, columns=["ticker", "asset_class_key", "asset_class", "role"]).set_index("ticker")
