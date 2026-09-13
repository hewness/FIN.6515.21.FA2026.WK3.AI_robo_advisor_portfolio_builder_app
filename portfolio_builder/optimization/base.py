"""Strategy interface for allocation approaches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inputs import MarketInputs
    from .models import AllocationResult, InvestorProfile


class OptimizationError(Exception):
    """Raised when an allocation cannot be produced or violates portfolio constraints."""


class AllocationStrategy(ABC):
    """An allocation approach: turns an investor profile and market inputs into a portfolio.

    Implementations must return long-only weights that sum to 1; ``AllocationResult``
    enforces this. Register new strategies with ``PortfolioOptimizationEngine.register``.
    """

    name: str

    @abstractmethod
    def allocate(self, profile: InvestorProfile, inputs: MarketInputs) -> AllocationResult:
        ...
