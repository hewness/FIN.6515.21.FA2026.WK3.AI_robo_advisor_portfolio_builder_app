"""Portfolio optimization engine: a registry of allocation strategies fed by the market data cache."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..market_data import MarketDataService, get_market_data_service
from .base import AllocationStrategy
from .inputs import DEFAULT_RISK_FREE_RATE, MarketInputs, build_market_inputs
from .mean_variance import MeanVarianceStrategy
from .models import AllocationResult, InvestorProfile
from .rule_based import RuleBasedStrategy


def default_strategies() -> list[AllocationStrategy]:
    return [RuleBasedStrategy(), MeanVarianceStrategy()]


class PortfolioOptimizationEngine:
    """Runs registered allocation strategies against inputs estimated from cached market data.

    Example::

        engine = get_optimization_engine()
        profile = InvestorProfile(age=40, risk_tolerance=6)
        result = engine.optimize("mean_variance", profile)
        result.weights, result.metrics, result.frontier.points
    """

    def __init__(
        self,
        market_data: MarketDataService | None = None,
        strategies: Iterable[AllocationStrategy] | None = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        frequency: str = "monthly",
        lookback_years: float | None = None,
    ) -> None:
        self.market_data = market_data or get_market_data_service()
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.lookback_years = lookback_years
        self._strategies: dict[str, AllocationStrategy] = {}
        self._inputs_cache: dict[tuple, MarketInputs] = {}
        for strategy in default_strategies() if strategies is None else strategies:
            self.register(strategy)

    # -- strategies --------------------------------------------------------
    def register(self, strategy: AllocationStrategy, replace: bool = False) -> None:
        if not isinstance(strategy, AllocationStrategy):
            raise TypeError("strategy must be an AllocationStrategy")
        if strategy.name in self._strategies and not replace:
            raise ValueError(f"Strategy {strategy.name!r} is already registered (pass replace=True)")
        self._strategies[strategy.name] = strategy

    def available_methods(self) -> list[str]:
        return list(self._strategies)

    def get_strategy(self, method: str) -> AllocationStrategy:
        try:
            return self._strategies[method]
        except KeyError:
            raise KeyError(f"Unknown method {method!r}; available: {self.available_methods()}") from None

    # -- inputs ------------------------------------------------------------
    def market_inputs(
        self,
        tickers: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> MarketInputs:
        """Annualized expected returns / covariance for ``tickers`` (default: the universe), memoized."""
        names = tuple(self.market_data.default_tickers if tickers is None else [t.upper() for t in tickers])
        key = (names, self.frequency, start, end, self.lookback_years, self.risk_free_rate)
        if key not in self._inputs_cache:
            self._inputs_cache[key] = build_market_inputs(
                self.market_data,
                list(names),
                frequency=self.frequency,
                start=start,
                end=end,
                lookback_years=self.lookback_years,
                risk_free_rate=self.risk_free_rate,
            )
        return self._inputs_cache[key]

    def clear_inputs_cache(self) -> None:
        """Forget memoized inputs, e.g. after refreshing market data."""
        self._inputs_cache.clear()

    # -- optimization ------------------------------------------------------
    def optimize(
        self,
        method: str,
        profile: InvestorProfile,
        *,
        tickers: Iterable[str] | None = None,
        inputs: MarketInputs | None = None,
        **strategy_options: Any,
    ) -> AllocationResult:
        """Allocate with a registered strategy.

        ``strategy_options`` build a one-off strategy of the same type, e.g.
        ``optimize("mean_variance", profile, objective="max_sharpe")``.
        """
        strategy = self.get_strategy(method)
        if strategy_options:
            strategy = type(strategy)(**strategy_options)
        return strategy.allocate(profile, inputs or self.market_inputs(tickers))

    def compare(self, profile: InvestorProfile, *, tickers: Iterable[str] | None = None) -> dict[str, AllocationResult]:
        """Run every registered strategy on the same inputs."""
        inputs = self.market_inputs(tickers)
        return {name: strategy.allocate(profile, inputs) for name, strategy in self._strategies.items()}


def get_optimization_engine(**kwargs: Any) -> PortfolioOptimizationEngine:
    """Engine backed by the default market data cache (see ``get_market_data_service``)."""
    return PortfolioOptimizationEngine(**kwargs)
