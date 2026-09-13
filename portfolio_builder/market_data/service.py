"""Cache-aside market data service: the interface used by optimization and backtesting."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..universe import get_tickers, universe_frame
from .cache import ParquetCache, normalize_ticker
from .connectors.base import HISTORY_COLUMNS, MarketDataConnector, MarketDataError

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "market_data"
CACHE_DIR_ENV = "MARKET_DATA_CACHE_DIR"

Frequency = Literal["daily", "weekly", "monthly"]
_RESAMPLE_RULES = {"weekly": "W-FRI", "monthly": "ME"}

DateLike = str | date | pd.Timestamp | None


class MarketDataService:
    """Serves market data from the local cache, downloading through the connector on a miss.

    Methods that take ``tickers`` default to the investment universe.
    """

    def __init__(
        self,
        connector: MarketDataConnector,
        cache: ParquetCache,
        default_tickers: Iterable[str] | None = None,
    ) -> None:
        self.connector = connector
        self.cache = cache
        self.default_tickers = [normalize_ticker(t) for t in (default_tickers or get_tickers())]

    # -- single ticker -----------------------------------------------------
    def get_history(self, ticker: str, refresh: bool = False) -> pd.DataFrame:
        """Full daily history for one ticker (normalized schema)."""
        t = normalize_ticker(ticker)
        if not refresh and self.cache.has_history(t):
            logger.info("%s: cache hit", t)
            return self.cache.read_history(t)
        return self._download_history(t)

    def get_info(self, ticker: str, refresh: bool = False) -> dict[str, Any]:
        """Descriptive fields for one ticker (name, category, expense ratio, ...)."""
        t = normalize_ticker(ticker)
        if not refresh and self.cache.has_info(t):
            return self.cache.read_info(t)
        info = self.connector.fetch_info(t)
        if info or not self.cache.has_info(t):
            self.cache.write_info(t, info)
            return info
        logger.warning("%s: info download returned nothing; keeping cached info", t)
        return self.cache.read_info(t)

    # -- panels for optimization / backtesting -----------------------------
    def get_prices(
        self,
        tickers: Iterable[str] | None = None,
        field: str = "adj_close",
        start: DateLike = None,
        end: DateLike = None,
        align: Literal["inner", "outer"] = "inner",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Wide frame of one history field (dates x tickers).

        ``align="inner"`` keeps only dates where every ticker has a value (use for
        covariance estimation); ``"outer"`` keeps all dates with NaN gaps.
        """
        if field not in HISTORY_COLUMNS:
            raise ValueError(f"Unknown field {field!r}; expected one of {HISTORY_COLUMNS}")
        if align not in ("inner", "outer"):
            raise ValueError("align must be 'inner' or 'outer'")
        names = self._resolve(tickers)
        frame = pd.concat({t: self.get_history(t, refresh=refresh)[field] for t in names}, axis=1)
        frame = frame.sort_index().loc[_ts(start) : _ts(end)]
        if align == "inner":
            frame = frame.dropna(how="any")
        frame.columns.name = "ticker"
        return frame

    def get_returns(
        self,
        tickers: Iterable[str] | None = None,
        frequency: Frequency = "daily",
        method: Literal["simple", "log"] = "simple",
        start: DateLike = None,
        end: DateLike = None,
        align: Literal["inner", "outer"] = "inner",
        refresh: bool = False,
        include_partial: bool = False,
    ) -> pd.DataFrame:
        """Periodic total returns from adjusted close prices (dates x tickers).

        Weekly and monthly periods are labeled by calendar period end. The final
        period is dropped when the data stops before its last business day, unless
        ``include_partial`` is set.
        """
        if frequency not in ("daily", "weekly", "monthly"):
            raise ValueError("frequency must be 'daily', 'weekly' or 'monthly'")
        if method not in ("simple", "log"):
            raise ValueError("method must be 'simple' or 'log'")
        prices = self.get_prices(tickers, "adj_close", start, end, align, refresh)
        if frequency != "daily" and not prices.empty:
            last_date = prices.index.max()
            prices = prices.resample(_RESAMPLE_RULES[frequency]).last()
            if not include_partial and last_date < pd.offsets.BDay().rollback(prices.index[-1]):
                prices = prices.iloc[:-1]
        if method == "simple":
            returns = prices.pct_change(fill_method=None)
        else:
            returns = np.log(prices / prices.shift(1))
        returns = returns.iloc[1:]
        return returns.dropna(how="any") if align == "inner" else returns.dropna(how="all")

    def get_dividends(
        self,
        tickers: Iterable[str] | None = None,
        start: DateLike = None,
        end: DateLike = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Wide frame of per-share cash dividends (0 on non-payment days)."""
        return self.get_prices(tickers, "dividends", start, end, align="outer", refresh=refresh).fillna(0.0)

    def get_universe_metadata(self, refresh: bool = False) -> pd.DataFrame:
        """Universe table (asset class, role) joined with cached descriptive info."""
        frame = universe_frame()
        info = pd.DataFrame.from_dict(
            {t: self.get_info(t, refresh=refresh) for t in frame.index}, orient="index"
        )
        return frame.join(info)

    # -- cache management --------------------------------------------------
    def ensure_cached(self, tickers: Iterable[str] | None = None) -> dict[str, str]:
        """Download history and info only for tickers not already on disk."""
        results: dict[str, str] = {}
        for t in self._resolve(tickers):
            if self.cache.has_history(t) and self.cache.has_info(t):
                results[t] = "cached"
                continue
            try:
                if not self.cache.has_history(t):
                    self._download_history(t)
                self.get_info(t)
                results[t] = "downloaded"
            except MarketDataError as exc:
                logger.error("%s: %s", t, exc)
                results[t] = f"failed: {exc}"
        return results

    def refresh(self, tickers: Iterable[str] | None = None) -> dict[str, str]:
        """Force re-download. On failure, existing cached data is left untouched."""
        results: dict[str, str] = {}
        for t in self._resolve(tickers):
            try:
                self._download_history(t)
                self.get_info(t, refresh=True)
                results[t] = "refreshed"
            except MarketDataError as exc:
                logger.error("%s: refresh failed, keeping cached data: %s", t, exc)
                results[t] = f"failed: {exc}"
        return results

    def cache_status(self, tickers: Iterable[str] | None = None) -> pd.DataFrame:
        names = self._resolve(tickers)
        status = self.cache.status(names)
        meta = universe_frame()[["asset_class"]]
        return meta.reindex(status.index).join(status)

    # -- internals ---------------------------------------------------------
    def _download_history(self, ticker: str) -> pd.DataFrame:
        logger.info("%s: downloading from %s", ticker, self.connector.name)
        df = self.connector.fetch_history(ticker)
        self.cache.write_history(
            ticker, df, source=self.connector.name, source_version=getattr(self.connector, "version", None)
        )
        return self.cache.read_history(ticker)

    def _resolve(self, tickers: Iterable[str] | None) -> list[str]:
        if tickers is None:
            return list(self.default_tickers)
        if isinstance(tickers, str):
            tickers = [tickers]
        names = list(dict.fromkeys(normalize_ticker(t) for t in tickers))
        if not names:
            raise ValueError("No tickers given")
        return names


def get_market_data_service(
    cache_dir: str | os.PathLike[str] | None = None,
    connector: MarketDataConnector | None = None,
) -> MarketDataService:
    """Build a service using ``cache_dir``, ``$MARKET_DATA_CACHE_DIR``, or ``data/market_data``."""
    root = Path(cache_dir or os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR)
    if connector is None:
        from .connectors.yfinance_connector import YFinanceConnector

        connector = YFinanceConnector()
    return MarketDataService(connector=connector, cache=ParquetCache(root))


def _ts(value: DateLike) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value)
