"""Portfolio optimization engine.

Typical use::

    from portfolio_builder.optimization import InvestorProfile, get_optimization_engine

    engine = get_optimization_engine()                    # cash return / risk-free rate defaults to 0%
    profile = InvestorProfile(age=40, risk_tolerance=6)
    rule = engine.optimize("rule_based", profile)
    mvo = engine.optimize("mean_variance", profile)       # includes mvo.frontier
    research = engine.optimize("research_informed", InvestorProfile(age=68, risk_tolerance=6.5, retirement_income=30_000,
                                                                    financial_wealth=500_000))
    engine.optimize("mean_variance", profile, objective="max_sharpe")
"""

from .base import AllocationStrategy, OptimizationError
from .engine import PortfolioOptimizationEngine, get_optimization_engine
from .inputs import MarketInputs, build_market_inputs, portfolio_metrics
from .mean_variance import MeanVarianceStrategy, efficient_frontier
from .models import CASH, AllocationResult, EfficientFrontier, InvestorProfile, Portfolio, PortfolioMetrics
from .research_informed import (
    PracticalFinanceAssumptions,
    ResearchEquity,
    ResearchInformedConfig,
    ResearchInformedStrategy,
    human_capital,
    merton_share,
    research_equity_share,
    risk_aversion,
)
from .rule_based import RuleBasedConfig, RuleBasedStrategy

__all__ = [
    "CASH",
    "AllocationResult",
    "AllocationStrategy",
    "EfficientFrontier",
    "InvestorProfile",
    "MarketInputs",
    "MeanVarianceStrategy",
    "OptimizationError",
    "Portfolio",
    "PortfolioMetrics",
    "PortfolioOptimizationEngine",
    "PracticalFinanceAssumptions",
    "ResearchEquity",
    "ResearchInformedConfig",
    "ResearchInformedStrategy",
    "RuleBasedConfig",
    "RuleBasedStrategy",
    "build_market_inputs",
    "efficient_frontier",
    "get_optimization_engine",
    "human_capital",
    "merton_share",
    "portfolio_metrics",
    "research_equity_share",
    "risk_aversion",
]
