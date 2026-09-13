"""Market data connector backed by Yahoo Finance via the ``yfinance`` library."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date
from typing import Any, TypeVar

import pandas as pd
import yfinance as yf

from .base import MarketDataError, validate_history

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
    "Capital Gains": "capital_gains",
}

# Subset of Ticker.info that is useful for portfolio construction and JSON-safe.
# yfinance reports netExpenseRatio in percent (0.03 == 0.03%).
INFO_FIELDS: tuple[str, ...] = (
    "longName",
    "quoteType",
    "category",
    "fundFamily",
    "currency",
    "netExpenseRatio",
    "yield",
    "totalAssets",
    "fundInceptionDate",
)


class YFinanceConnector:
    name = "yfinance"

    def __init__(self, max_retries: int = 3, backoff_seconds: float = 2.0) -> None:
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    @property
    def version(self) -> str:
        return getattr(yf, "__version__", "unknown")

    def fetch_history(self, ticker: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"auto_adjust": False, "actions": True}
        if start is None and end is None:
            kwargs["period"] = "max"
        else:
            kwargs.update(start=start, end=end)

        def call() -> pd.DataFrame:
            raw = yf.Ticker(ticker).history(**kwargs)
            # Validation runs inside the retry loop: yfinance can return an empty
            # frame on transient failures instead of raising.
            return validate_history(raw.rename(columns=_RENAME), ticker)

        return self._with_retries(call, f"history for {ticker}")

    def fetch_info(self, ticker: str) -> dict[str, Any]:
        try:
            info = self._with_retries(lambda: yf.Ticker(ticker).info or {}, f"info for {ticker}")
        except MarketDataError as exc:
            logger.warning("%s", exc)
            return {}
        return {k: info[k] for k in INFO_FIELDS if info.get(k) is not None}

    def _with_retries(self, fn: Callable[[], T], what: str) -> T:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # yfinance raises assorted network / rate-limit errors
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.backoff_seconds * 2 ** (attempt - 1)
                    logger.warning(
                        "Fetching %s failed (attempt %d/%d): %s; retrying in %.1fs",
                        what, attempt, self.max_retries, exc, delay,
                    )
                    time.sleep(delay)
        raise MarketDataError(f"Failed to fetch {what} after {self.max_retries} attempts: {last_exc}") from last_exc
