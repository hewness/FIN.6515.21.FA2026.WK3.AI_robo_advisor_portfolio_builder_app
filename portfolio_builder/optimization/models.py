"""Investor profile and allocation result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..universe import get_asset_class, universe_frame
from .base import OptimizationError

# Placeholder ticker for the Cash and Money Markets asset class (no tradable ticker).
CASH = "CASH"

MIN_AGE, MAX_AGE = 18, 100
MIN_RISK, MAX_RISK = 1.0, 10.0
WEIGHT_TOLERANCE = 1e-6
# Retirement income as a share of final labor income when none is given (Practical Finance's low case).
DEFAULT_REPLACEMENT_RATE = 0.40


@dataclass(frozen=True)
class InvestorProfile:
    age: int
    risk_tolerance: float  # 1 (most conservative) .. 10 (most aggressive)
    # Income and wealth, used by the research-informed model (human capital); other models ignore them.
    annual_income: float = 0.0  # current real labor income per year (0 if retired)
    retirement_income: float | None = None  # Social Security + pensions per year; None = 40% of annual income
    retirement_age: int = 67  # first age at which retirement income replaces labor income
    financial_wealth: float | None = None  # investable financial wealth being allocated

    def __post_init__(self) -> None:
        if not MIN_AGE <= self.age <= MAX_AGE:
            raise ValueError(f"age must be between {MIN_AGE} and {MAX_AGE}, got {self.age}")
        if not MIN_RISK <= self.risk_tolerance <= MAX_RISK:
            raise ValueError(f"risk_tolerance must be between 1 and 10, got {self.risk_tolerance}")
        if self.annual_income < 0 or (self.retirement_income is not None and self.retirement_income < 0):
            raise ValueError("annual_income and retirement_income must be non-negative")
        if self.financial_wealth is not None and self.financial_wealth <= 0:
            raise ValueError(f"financial_wealth must be positive, got {self.financial_wealth}")

    @property
    def resolved_retirement_income(self) -> float:
        """Retirement income, defaulting to ``DEFAULT_REPLACEMENT_RATE`` of annual income."""
        if self.retirement_income is not None:
            return float(self.retirement_income)
        return DEFAULT_REPLACEMENT_RATE * self.annual_income

    @property
    def risk_fraction(self) -> float:
        """Risk tolerance mapped linearly onto [0, 1]."""
        return (self.risk_tolerance - MIN_RISK) / (MAX_RISK - MIN_RISK)


@dataclass(frozen=True)
class PortfolioMetrics:
    """Annualized expected return, volatility and Sharpe ratio."""

    expected_return: float
    volatility: float
    sharpe_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {"expected_return": self.expected_return, "volatility": self.volatility, "sharpe_ratio": self.sharpe_ratio}


def clean_weights(weights: pd.Series | dict[str, float]) -> pd.Series:
    """Validate long-only, fully-invested weights; clip float noise and renormalize.

    Raises ``OptimizationError`` on negative weights, non-finite values, or a sum away from 1.
    """
    w = pd.Series(weights, dtype="float64")
    if w.empty:
        raise OptimizationError("Portfolio has no weights")
    if not np.isfinite(w.to_numpy()).all():
        raise OptimizationError(f"Portfolio weights must be finite: {w.to_dict()}")
    if (w < -WEIGHT_TOLERANCE).any():
        raise OptimizationError(f"Short positions are not allowed: {w[w < 0].to_dict()}")
    if abs(w.sum() - 1.0) > WEIGHT_TOLERANCE:
        raise OptimizationError(f"Portfolio weights must sum to 1, got {w.sum():.8f}")
    w = w.clip(lower=0.0)
    w[w < 1e-10] = 0.0
    return w / w.sum()


@dataclass
class Portfolio:
    """A named set of weights with its metrics, e.g. a reference point on the frontier."""

    weights: pd.Series
    metrics: PortfolioMetrics

    def __post_init__(self) -> None:
        self.weights = clean_weights(self.weights)


@dataclass
class EfficientFrontier:
    """Frontier points (expected_return, volatility, sharpe_ratio) and the weights of each point."""

    points: pd.DataFrame
    weights: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.points.join(self.weights)

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class AllocationResult:
    method: str
    profile: InvestorProfile
    weights: pd.Series
    metrics: PortfolioMetrics
    frontier: EfficientFrontier | None = None
    reference_portfolios: dict[str, Portfolio] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = clean_weights(self.weights)
        self.weights.name = "weight"

    @property
    def asset_class_weights(self) -> pd.Series:
        """Weights aggregated by asset class name (CASH maps to Cash and Money Markets)."""
        return group_by_asset_class(self.weights)

    def to_frame(self, include_zero: bool = False) -> pd.DataFrame:
        """Weights with asset class and role per ticker, sorted by weight."""
        weights = self.weights if include_zero else self.weights[self.weights > 0]
        meta = _ticker_metadata()
        frame = meta.reindex(weights.index).assign(weight=weights)
        return frame.sort_values("weight", ascending=False)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"{self.method} | age {self.profile.age}, risk tolerance {self.profile.risk_tolerance:g}",
            f"expected return {m.expected_return:.2%} | volatility {m.volatility:.2%} | Sharpe {m.sharpe_ratio:.2f}",
        ]
        lines += [f"  {t:6} {w:7.2%}" for t, w in self.weights[self.weights > 0].sort_values(ascending=False).items()]
        return "\n".join(lines)


def group_by_asset_class(weights: pd.Series) -> pd.Series:
    meta = _ticker_metadata()
    classes = meta["asset_class"].reindex(weights.index)
    grouped = weights.groupby(classes, sort=False).sum()
    grouped.index.name = "asset_class"
    return grouped[grouped > 0]


def _ticker_metadata() -> pd.DataFrame:
    frame = universe_frame()[["asset_class", "role"]]
    cash = get_asset_class("cash")
    frame.loc[CASH] = [cash.name, cash.role]
    return frame
