"""Service layer between a UI and the portfolio optimization engine.

Typical use from a UI handler::

    from portfolio_builder.service import recommend_portfolio, InputValidationError

    response = recommend_portfolio(risk_tolerance="moderate", horizon_years=20, initial_investment=100_000,
                                   monthly_contribution=500, goal="retirement", age=40)
    response.holdings_frame("rule_based")       # table
    response.frontier_frame()                   # chart data
    response.to_dict()                          # JSON
"""

from .frontier_position import locate_on_frontier
from .portfolio_service import (
    InputValidationError,
    PortfolioService,
    PortfolioServiceError,
    get_portfolio_service,
    recommend_portfolio,
)
from .projection import ProjectionConfig, project_portfolio_value
from .risk import RISK_LABELS, RiskLevel, horizon_adjusted_risk, resolve_risk_tolerance, risk_band
from .schemas import (
    EfficientFrontierData,
    FinancialGoal,
    FrontierPosition,
    Holding,
    PortfolioRecommendation,
    PortfolioRequest,
    PortfolioResponse,
    Projection,
    get_form_options,
)

__all__ = [
    "RISK_LABELS",
    "EfficientFrontierData",
    "FinancialGoal",
    "FrontierPosition",
    "Holding",
    "InputValidationError",
    "PortfolioRecommendation",
    "PortfolioRequest",
    "PortfolioResponse",
    "PortfolioService",
    "PortfolioServiceError",
    "Projection",
    "ProjectionConfig",
    "RiskLevel",
    "get_form_options",
    "get_portfolio_service",
    "horizon_adjusted_risk",
    "locate_on_frontier",
    "project_portfolio_value",
    "recommend_portfolio",
    "resolve_risk_tolerance",
    "risk_band",
]
