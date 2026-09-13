"""Rule-based allocation: equity share from a glide path (default: 110 - age), adjusted by risk tolerance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..universe import get_asset_class
from .base import AllocationStrategy, OptimizationError
from .glide_path import GlidePath, LinearGlidePath, equity_target
from .inputs import MarketInputs, portfolio_metrics
from .models import CASH, AllocationResult, InvestorProfile


@dataclass(frozen=True)
class RuleBasedConfig:
    base: float = 110.0
    # Risk tolerance 1 shifts equity by -max_risk_shift points, 10 by +max_risk_shift, linear between.
    max_risk_shift: float = 20.0
    min_equity: float = 0.10
    max_equity: float = 1.00
    # Split of the equity portion across asset classes (must sum to 1).
    equity_sleeve: dict[str, float] = field(
        default_factory=lambda: {
            "us_large_cap": 0.55,
            "intl_developed": 0.25,
            "emerging_markets": 0.10,
            "real_estate": 0.10,
        }
    )
    # Split of the remaining defensive portion across asset classes (must sum to 1).
    defensive_sleeve: dict[str, float] = field(
        default_factory=lambda: {"us_aggregate_bonds": 0.70, "tips": 0.20, "cash": 0.10}
    )
    # Equity share by age before the risk shift; None means the linear ``base - age`` rule.
    glide_path: GlidePath | None = None

    @property
    def path(self) -> GlidePath:
        return self.glide_path if self.glide_path is not None else LinearGlidePath(self.base)

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_equity <= self.max_equity <= 1.0:
            raise ValueError("Require 0 <= min_equity <= max_equity <= 1")
        for label, sleeve in (("equity_sleeve", self.equity_sleeve), ("defensive_sleeve", self.defensive_sleeve)):
            if any(v < 0 for v in sleeve.values()) or not math.isclose(sum(sleeve.values()), 1.0, abs_tol=1e-9):
                raise ValueError(f"{label} weights must be non-negative and sum to 1")
            for key in sleeve:
                get_asset_class(key)  # raises KeyError for unknown asset classes


class RuleBasedStrategy(AllocationStrategy):
    name = "rule_based"

    def __init__(self, config: RuleBasedConfig | None = None) -> None:
        self.config = config or RuleBasedConfig()

    def equity_fraction(self, profile: InvestorProfile) -> tuple[float, float]:
        """Return (equity fraction after clamping, risk shift in percentage points)."""
        cfg = self.config
        return equity_target(cfg.path, profile, cfg.max_risk_shift, cfg.min_equity, cfg.max_equity)

    def allocate(self, profile: InvestorProfile, inputs: MarketInputs) -> AllocationResult:
        equity, shift = self.equity_fraction(profile)
        class_weights: dict[str, float] = {}
        for key, share in self.config.equity_sleeve.items():
            class_weights[key] = class_weights.get(key, 0.0) + equity * share
        for key, share in self.config.defensive_sleeve.items():
            class_weights[key] = class_weights.get(key, 0.0) + (1.0 - equity) * share

        weights: dict[str, float] = {}
        ticker_order: dict[str, list[str]] = {}
        for key, class_weight in class_weights.items():
            if class_weight <= 0:
                continue
            if key == "cash":
                weights[CASH] = weights.get(CASH, 0.0) + class_weight
                continue
            order, split = select_tickers(key, profile.risk_fraction, inputs)
            ticker_order[key] = order
            for ticker, fraction in split.items():
                weights[ticker] = weights.get(ticker, 0.0) + class_weight * fraction

        series = pd.Series(weights, dtype="float64")
        return AllocationResult(
            method=self.name,
            profile=profile,
            weights=series,
            metrics=portfolio_metrics(series, inputs),
            details={
                "equity_pct": equity,
                "glide_path": self.config.path.name,
                "base_equity_pct": self.config.path.base_equity(profile.age),
                "risk_shift_points": shift,
                "asset_class_weights": class_weights,
                "ticker_order_by_volatility": ticker_order,
                "estimation_window": inputs.estimation_window,
                "risk_free_rate": inputs.risk_free_rate,
            },
        )


def select_tickers(asset_class_key: str, risk_fraction: float, inputs: MarketInputs) -> tuple[list[str], dict[str, float]]:
    """Blend an asset class's tickers by risk tolerance.

    Tickers are ordered from lowest to highest historical volatility; the allocation sits at
    position ``risk_fraction * (n - 1)`` along that ordering, split linearly between the two
    neighbouring tickers. Conservative clients get the calmest fund, aggressive clients the
    most volatile one.
    """
    tickers = [t for t in get_asset_class(asset_class_key).tickers if t in inputs.tickers]
    if not tickers:
        raise OptimizationError(f"No market inputs for any ticker in asset class {asset_class_key!r}")
    vols = inputs.volatilities
    order = sorted(tickers, key=lambda t: (vols[t], t))
    position = min(max(risk_fraction, 0.0), 1.0) * (len(order) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(order) - 1)
    frac = position - lower
    split = {order[lower]: 1.0 - frac}
    if upper != lower and frac > 0:
        split[order[upper]] = frac
    return order, split
