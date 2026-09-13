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
    AssetClass("us_large_cap", "US large-cap stocks", "Growth, domestic equity exposure", ("SPY", "VTI")),
    AssetClass(
        "intl_developed",
        "International developed stocks",
        "Diversification, international exposure",
        ("EFA", "VXUS"),
    ),
    AssetClass("emerging_markets", "Emerging market stocks", "Higher growth potential, higher risk", ("EEM", "VWO")),
    AssetClass("us_aggregate_bonds", "US Aggregate bonds", "Stability, income", ("AGG", "BND")),
    AssetClass("tips", "Treasury inflation-protected securities", "Inflation Hedge", ("TIP", "VTIP")),
    AssetClass("real_estate", "Real estate (REITs)", "Real asset diversification", ("VNQ",)),
    # Cash has no ticker; it is modeled outside the market data feed.
    AssetClass("cash", "Cash and Money Markets", "Liquidity, capital preservation"),
)

_BY_KEY: dict[str, AssetClass] = {ac.key: ac for ac in ASSET_CLASSES}
_BY_TICKER: dict[str, AssetClass] = {}
for _ac in ASSET_CLASSES:
    for _ticker in _ac.tickers:
        if _ticker in _BY_TICKER:
            raise ValueError(f"Ticker {_ticker} is assigned to more than one asset class")
        _BY_TICKER[_ticker] = _ac


def get_asset_classes() -> tuple[AssetClass, ...]:
    return ASSET_CLASSES


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
