"""Connector interface and the normalized history schema shared by cache and service."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, runtime_checkable

import pandas as pd

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close")
HISTORY_COLUMNS: tuple[str, ...] = (*PRICE_COLUMNS, "volume", "dividends", "stock_splits", "capital_gains")
INDEX_NAME = "date"


class MarketDataError(Exception):
    """Raised when market data cannot be retrieved or fails validation."""


@runtime_checkable
class MarketDataConnector(Protocol):
    """A source of daily history and descriptive info for a single ticker."""

    name: str

    def fetch_history(self, ticker: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Return daily history in the normalized schema (see ``validate_history``)."""
        ...

    def fetch_info(self, ticker: str) -> dict[str, Any]:
        """Return JSON-serializable descriptive fields; may be empty."""
        ...


def validate_history(df: pd.DataFrame | None, ticker: str) -> pd.DataFrame:
    """Coerce a history frame to the normalized schema or raise ``MarketDataError``.

    Schema: tz-naive, sorted, unique ``DatetimeIndex`` named ``date``; columns
    ``HISTORY_COLUMNS`` as float64 except ``volume`` (int64).
    """
    if df is None or df.empty:
        raise MarketDataError(f"No history returned for {ticker}")
    missing = [c for c in (*PRICE_COLUMNS, "volume") if c not in df.columns]
    if missing:
        raise MarketDataError(f"History for {ticker} is missing columns: {missing}")

    out = df.copy()
    for col in ("dividends", "stock_splits", "capital_gains"):
        if col not in out.columns:
            out[col] = 0.0
    out = out[list(HISTORY_COLUMNS)]

    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize().as_unit("ns").rename(INDEX_NAME)

    out = out.dropna(subset=list(PRICE_COLUMNS), how="all")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise MarketDataError(f"No usable price rows for {ticker}")

    float_cols = [c for c in HISTORY_COLUMNS if c != "volume"]
    out[float_cols] = out[float_cols].astype("float64")
    out[["dividends", "stock_splits", "capital_gains"]] = out[["dividends", "stock_splits", "capital_gains"]].fillna(0.0)
    out["volume"] = out["volume"].fillna(0).astype("int64")
    return out
