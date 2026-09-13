"""Request and response models for the portfolio service (pydantic, JSON-safe)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .risk import MAX_RISK, MIN_RISK, RISK_LABELS, RiskLevel, resolve_risk_tolerance

Method = Literal["rule_based", "mean_variance"]
RebalanceFrequency = Literal["monthly", "quarterly", "annual"]


class FinancialGoal(str, Enum):
    RETIREMENT = "retirement"
    HOME_PURCHASE = "home_purchase"
    EDUCATION = "education"
    GENERAL_WEALTH = "general_wealth"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").capitalize()

    @property
    def default_target(self) -> float:
        return GOAL_TARGET_DEFAULTS[self]


GOAL_TARGET_DEFAULTS: dict[FinancialGoal, float] = {
    FinancialGoal.RETIREMENT: 1_500_000.0,
    FinancialGoal.HOME_PURCHASE: 150_000.0,
    FinancialGoal.EDUCATION: 200_000.0,
    FinancialGoal.GENERAL_WEALTH: 1_000_000.0,
}


# -- request -------------------------------------------------------------------
class PortfolioRequest(BaseModel):
    """Client inputs collected by the UI."""

    model_config = ConfigDict(extra="forbid")

    risk_tolerance: float | RiskLevel = Field(
        default=5.5,
        title="Risk tolerance",
        description="Drives equity vs. bond allocation: 1 (conservative) to 10 (aggressive), or a label.",
        json_schema_extra={"widget": "slider_or_select", "minimum": MIN_RISK, "maximum": MAX_RISK, "step": 0.5},
    )
    horizon_years: int = Field(
        default=25, ge=1, le=30, title="Investment horizon (years)",
        description="Longer horizons allow more risk.", json_schema_extra={"widget": "slider", "step": 1},
    )
    initial_investment: float = Field(
        default=50_000, ge=1_000, le=10_000_000, title="Initial investment ($)",
        description="Starting portfolio value.", json_schema_extra={"widget": "number", "step": 1_000},
    )
    monthly_contribution: float = Field(
        default=1_000, ge=0, le=50_000, title="Monthly contribution ($)",
        description="Ongoing savings added each month.", json_schema_extra={"widget": "number", "step": 100},
    )
    goal: FinancialGoal = Field(
        default=FinancialGoal.RETIREMENT, title="Financial goal",
        description="Context for the recommendation.", json_schema_extra={"widget": "select"},
    )
    age: int = Field(
        default=40, ge=18, le=80, title="Age",
        description="Used in lifecycle allocation rules.", json_schema_extra={"widget": "number", "step": 1},
    )
    target_amount: float | None = Field(
        default=None, ge=1_000, le=100_000_000, title="Goal target ($)",
        description="Amount you want to reach by the end of the horizon (defaults by goal).",
        json_schema_extra={"widget": "number", "step": 10_000},
    )
    backtest_years: int = Field(
        default=15, ge=10, le=20, title="Backtest lookback (years)",
        description="How far back to test the allocation against the S&P 500.",
        json_schema_extra={"widget": "slider", "step": 1},
    )
    rebalance: RebalanceFrequency = Field(
        default="quarterly", title="Rebalancing",
        description="How often the backtest resets holdings to target weights.",
        json_schema_extra={"widget": "radio"},
    )

    @field_validator("risk_tolerance", mode="before")
    @classmethod
    def _parse_risk(cls, value: Any) -> float | RiskLevel:
        if isinstance(value, RiskLevel):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in RiskLevel._value2member_map_:
                return RiskLevel(text)
        try:
            return resolve_risk_tolerance(value)
        except (TypeError, ValueError):
            labels = ", ".join(level.label for level in RiskLevel)
            raise ValueError(f"Risk tolerance must be a number from 1 to 10 or one of: {labels}") from None

    @field_validator("goal", mode="before")
    @classmethod
    def _parse_goal(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_").replace(" ", "_")
        return value

    @field_validator("rebalance", mode="before")
    @classmethod
    def _parse_rebalance(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("target_amount", mode="before")
    @classmethod
    def _blank_target(cls, value: Any) -> Any:
        return None if value in ("", None) else value

    @property
    def risk_score(self) -> float:
        """Risk tolerance as a 1-10 number (labels resolved)."""
        return resolve_risk_tolerance(self.risk_tolerance)

    @property
    def resolved_target(self) -> float:
        """The goal target, or the goal's default when none was entered."""
        return self.target_amount if self.target_amount is not None else self.goal.default_target


def get_form_options() -> dict[str, dict[str, Any]]:
    """Widget metadata for each request field, derived from ``PortfolioRequest``."""
    schema = PortfolioRequest.model_json_schema()["properties"]
    options: dict[str, dict[str, Any]] = {}
    for name, field in PortfolioRequest.model_fields.items():
        extra = dict(field.json_schema_extra or {})
        prop = schema[name]
        default = field.default.value if isinstance(field.default, Enum) else field.default
        entry: dict[str, Any] = {
            "label": field.title,
            "help": field.description,
            "widget": extra.pop("widget", None),
            "minimum": extra.pop("minimum", prop.get("minimum")),
            "maximum": extra.pop("maximum", prop.get("maximum")),
            "step": extra.pop("step", None),
            "default": default,
        }
        if name == "risk_tolerance":
            entry["choices"] = [{"value": level.value, "label": level.label, "score": score}
                                for level, score in RISK_LABELS.items()]
        elif name == "goal":
            entry["choices"] = [{"value": goal.value, "label": goal.label, "default_target": goal.default_target}
                                for goal in FinancialGoal]
        elif name == "target_amount":
            entry["default"] = PortfolioRequest.model_fields["goal"].default.default_target
            entry["minimum"], entry["maximum"] = 1_000, 100_000_000
        elif name == "rebalance":
            entry["choices"] = [{"value": v, "label": v.capitalize()} for v in ("monthly", "quarterly", "annual")]
        options[name] = entry
    return options


# -- response ------------------------------------------------------------------
class ProfileSummary(BaseModel):
    age: int
    goal: FinancialGoal
    goal_label: str
    horizon_years: int
    risk_tolerance_input: str
    risk_tolerance: float
    effective_risk_tolerance: float
    risk_band: str
    horizon_adjustment: str


class MarketDataSummary(BaseModel):
    estimation_start: str | None
    estimation_end: str | None
    observations: int
    frequency: str
    risk_free_rate: float
    data_as_of: str | None


class Holding(BaseModel):
    ticker: str
    name: str
    asset_class: str
    role: str
    weight: float
    amount: float


class AssetClassAllocation(BaseModel):
    asset_class: str
    role: str
    weight: float
    amount: float


class FrontierPosition(BaseModel):
    volatility: float
    expected_return: float
    frontier_return_at_volatility: float
    return_gap: float
    position: float
    nearest_point: int
    on_frontier: bool
    note: str | None = None


class ProjectionPoint(BaseModel):
    year: int
    total_contributed: float
    expected: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


class Projection(BaseModel):
    points: list[ProjectionPoint]
    final_expected: float
    final_p10: float
    final_p25: float
    final_p50: float
    final_p75: float
    final_p90: float
    total_contributed: float
    target_amount: float | None = None
    probability_of_meeting_target: float | None = None
    simulations: int
    seed: int | None


class PortfolioRecommendation(BaseModel):
    method: Method
    title: str
    description: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    holdings: list[Holding]
    asset_classes: list[AssetClassAllocation]
    frontier_position: FrontierPosition
    projection: Projection
    max_drawdown: float | None = None  # from the historical backtest
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def probability_of_meeting_target(self) -> float | None:
        return self.projection.probability_of_meeting_target


class FrontierPoint(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float]


class ReferencePortfolio(BaseModel):
    name: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float]


class PortfolioPoint(BaseModel):
    label: str
    expected_return: float
    volatility: float


class AssetClassPoint(BaseModel):
    """Risk/return of an asset class (equal-weight blend of its funds); Cash sits at the risk-free rate."""

    asset_class: str
    tickers: list[str]
    expected_return: float
    volatility: float


class EfficientFrontierData(BaseModel):
    points: list[FrontierPoint]
    reference_portfolios: list[ReferencePortfolio]
    client_point: PortfolioPoint
    rule_based_point: PortfolioPoint
    asset_class_points: list[AssetClassPoint] = Field(default_factory=list)


class BacktestMetricsData(BaseModel):
    label: str
    final_value: float
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_date: str | None
    best_year: float | None
    worst_year: float | None


class BacktestPoint(BaseModel):
    date: str
    rule_based: float
    mean_variance: float
    benchmark: float
    rule_based_drawdown: float
    mean_variance_drawdown: float
    benchmark_drawdown: float


class ProxyUsage(BaseModel):
    ticker: str
    proxy: str
    until: str


class BacktestData(BaseModel):
    start: str
    end: str
    years_requested: int
    years_covered: float
    rebalance: str
    benchmark: str
    benchmark_label: str
    initial_value: float
    points: list[BacktestPoint]
    metrics: dict[str, BacktestMetricsData]
    proxies_used: list[ProxyUsage]
    notes: list[str]


class PortfolioResponse(BaseModel):
    request: PortfolioRequest
    profile: ProfileSummary
    market_data: MarketDataSummary
    rule_based: PortfolioRecommendation
    mean_variance: PortfolioRecommendation
    efficient_frontier: EfficientFrontierData
    backtest: BacktestData
    notes: list[str]
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def recommendation(self, method: Method) -> PortfolioRecommendation:
        if method == "rule_based":
            return self.rule_based
        if method == "mean_variance":
            return self.mean_variance
        raise ValueError("method must be 'rule_based' or 'mean_variance'")

    # DataFrame helpers for UI tables and charts.
    def holdings_frame(self, method: Method) -> pd.DataFrame:
        return pd.DataFrame([h.model_dump() for h in self.recommendation(method).holdings],
                            columns=list(Holding.model_fields))

    def asset_class_frame(self, method: Method) -> pd.DataFrame:
        return pd.DataFrame([a.model_dump() for a in self.recommendation(method).asset_classes],
                            columns=list(AssetClassAllocation.model_fields))

    def projection_frame(self, method: Method) -> pd.DataFrame:
        return pd.DataFrame([p.model_dump() for p in self.recommendation(method).projection.points],
                            columns=list(ProjectionPoint.model_fields))

    def frontier_frame(self) -> pd.DataFrame:
        rows = [{"expected_return": p.expected_return, "volatility": p.volatility,
                 "sharpe_ratio": p.sharpe_ratio, **p.weights} for p in self.efficient_frontier.points]
        return pd.DataFrame(rows)

    def backtest_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([p.model_dump() for p in self.backtest.points], columns=list(BacktestPoint.model_fields))
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def comparison_frame(self) -> pd.DataFrame:
        rows = []
        for rec in (self.rule_based, self.mean_variance):
            rows.append({
                "method": rec.method,
                "title": rec.title,
                "expected_return": rec.expected_return,
                "volatility": rec.volatility,
                "sharpe_ratio": rec.sharpe_ratio,
                "max_drawdown": rec.max_drawdown,
                "probability_of_meeting_target": rec.probability_of_meeting_target,
                "frontier_return_gap": rec.frontier_position.return_gap,
                "frontier_position": rec.frontier_position.position,
                "projected_p10": rec.projection.final_p10,
                "projected_median": rec.projection.final_p50,
                "projected_p90": rec.projection.final_p90,
            })
        return pd.DataFrame(rows)
