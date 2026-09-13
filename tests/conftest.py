from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from portfolio_builder.market_data import MarketDataError, MarketDataService, ParquetCache


def make_history(start: str = "2020-01-01", periods: int = 300, seed: int = 0, price: float = 100.0) -> pd.DataFrame:
    """Synthetic business-day history in the normalized schema."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods, name="date")
    close = price * np.cumprod(1 + rng.normal(0.0003, 0.01, periods))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000, 10_000, periods),
            "dividends": 0.0,
            "stock_splits": 0.0,
            "capital_gains": 0.0,
        },
        index=idx,
    )


class FakeConnector:
    name = "fake"
    version = "0.0"

    def __init__(self, starts: dict[str, str] | None = None, fail: set[str] | None = None) -> None:
        self.starts = starts or {}
        self.fail = fail or set()
        self.history_calls: Counter[str] = Counter()
        self.info_calls: Counter[str] = Counter()
        self.generation = 0  # bump to make subsequent downloads return different prices

    def fetch_history(self, ticker, start=None, end=None):
        self.history_calls[ticker] += 1
        if ticker in self.fail:
            raise MarketDataError(f"boom {ticker}")
        seed = sum(map(ord, ticker)) + self.generation
        return make_history(start=self.starts.get(ticker, "2020-01-01"), seed=seed, price=100.0 + self.generation)

    def fetch_info(self, ticker):
        self.info_calls[ticker] += 1
        if ticker in self.fail:
            return {}
        return {"longName": f"{ticker} Fund", "netExpenseRatio": 0.05}


class LongHistoryConnector(FakeConnector):
    """Five years of daily data so monthly estimation has enough observations."""

    def fetch_history(self, ticker, start=None, end=None):
        self.history_calls[ticker] += 1
        seed = sum(map(ord, ticker))
        return make_history(start="2018-01-01", periods=1300, seed=seed)


@pytest.fixture
def cache(tmp_path) -> ParquetCache:
    return ParquetCache(tmp_path / "cache")


@pytest.fixture
def connector() -> FakeConnector:
    return FakeConnector()


@pytest.fixture
def service(connector, cache) -> MarketDataService:
    return MarketDataService(connector, cache)
