"""Estimate optimizer inputs (expected returns, covariance) from the market data cache."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..market_data import MarketDataService
from .base import OptimizationError
from .models import CASH, PortfolioMetrics

PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}
MIN_OBSERVATIONS = 24
DEFAULT_RISK_FREE_RATE = 0.04


@dataclass(frozen=True)
class MarketInputs:
    """Annualized expected returns and covariance for a set of tickers."""

    expected_returns: pd.Series
    covariance: pd.DataFrame
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    frequency: str = "monthly"
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    observations: int = 0

    def __post_init__(self) -> None:
        tickers = list(self.expected_returns.index)
        if list(self.covariance.index) != tickers or list(self.covariance.columns) != tickers:
            raise OptimizationError("Covariance must be indexed by the same tickers as expected returns")
        if self.expected_returns.isna().any() or self.covariance.isna().any().any():
            raise OptimizationError("Expected returns and covariance must not contain NaN")

    @property
    def tickers(self) -> list[str]:
        return list(self.expected_returns.index)

    @property
    def volatilities(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.covariance.to_numpy())), index=self.tickers, name="volatility")

    @property
    def correlation(self) -> pd.DataFrame:
        vol = self.volatilities.to_numpy()
        return self.covariance / np.outer(vol, vol)

    @property
    def estimation_window(self) -> dict[str, object]:
        return {
            "start": None if self.start is None else self.start.date().isoformat(),
            "end": None if self.end is None else self.end.date().isoformat(),
            "observations": self.observations,
            "frequency": self.frequency,
        }

    @classmethod
    def from_returns(cls, returns: pd.DataFrame, frequency: str = "monthly",
                     risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> MarketInputs:
        """Annualize sample mean and covariance of periodic returns."""
        if frequency not in PERIODS_PER_YEAR:
            raise ValueError(f"frequency must be one of {sorted(PERIODS_PER_YEAR)}")
        returns = returns.dropna(how="any")
        if len(returns) < MIN_OBSERVATIONS:
            raise OptimizationError(
                f"Need at least {MIN_OBSERVATIONS} {frequency} observations, got {len(returns)}"
            )
        periods = PERIODS_PER_YEAR[frequency]
        return cls(
            expected_returns=returns.mean() * periods,
            covariance=returns.cov() * periods,
            risk_free_rate=risk_free_rate,
            frequency=frequency,
            start=returns.index.min(),
            end=returns.index.max(),
            observations=len(returns),
        )


def build_market_inputs(
    service: MarketDataService,
    tickers: Iterable[str] | None = None,
    frequency: str = "monthly",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    lookback_years: float | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> MarketInputs:
    """Returns over the common window where every ticker has data, annualized."""
    returns = service.get_returns(tickers, frequency=frequency, start=start, end=end, align="inner")
    if lookback_years is not None and not returns.empty:
        cutoff = returns.index.max() - pd.DateOffset(months=round(lookback_years * 12))
        returns = returns[returns.index > cutoff]
    return MarketInputs.from_returns(returns, frequency=frequency, risk_free_rate=risk_free_rate)


def portfolio_metrics(weights: pd.Series, inputs: MarketInputs) -> PortfolioMetrics:
    """Annualized metrics; a CASH weight earns the risk-free rate with zero volatility."""
    cash = float(weights.get(CASH, 0.0))
    risky = weights.drop(CASH, errors="ignore")
    unknown = set(risky.index) - set(inputs.tickers)
    if unknown:
        raise OptimizationError(f"No market inputs for: {sorted(unknown)}")
    w = risky.reindex(inputs.tickers, fill_value=0.0).to_numpy()
    expected = float(w @ inputs.expected_returns.to_numpy()) + cash * inputs.risk_free_rate
    volatility = float(np.sqrt(max(w @ inputs.covariance.to_numpy() @ w, 0.0)))
    sharpe = (expected - inputs.risk_free_rate) / volatility if volatility > 1e-12 else 0.0
    return PortfolioMetrics(expected, volatility, sharpe)
