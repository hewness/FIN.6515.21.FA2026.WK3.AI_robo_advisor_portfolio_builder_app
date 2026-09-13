"""Service layer connecting a UI to the portfolio optimization engine."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..market_data import MarketDataError
from ..optimization import (
    CASH,
    AllocationResult,
    InvestorProfile,
    MarketInputs,
    OptimizationError,
    PortfolioOptimizationEngine,
    get_optimization_engine,
)
from ..universe import get_asset_class, get_asset_class_for_ticker
from .frontier_position import locate_on_frontier
from .projection import ProjectionConfig, project_portfolio_value
from .recommendations import build_notes
from .risk import RiskLevel, horizon_adjusted_risk, risk_band
from .schemas import (
    AssetClassAllocation,
    EfficientFrontierData,
    FrontierPoint,
    Holding,
    MarketDataSummary,
    PortfolioPoint,
    PortfolioRecommendation,
    PortfolioRequest,
    PortfolioResponse,
    ProfileSummary,
    ReferencePortfolio,
    get_form_options,
)

METHOD_TEXT = {
    "rule_based": (
        "Rule-based lifecycle portfolio",
        "Equity share = 110 - age, adjusted for risk tolerance; split across asset classes by fixed sleeves.",
    ),
    "mean_variance": (
        "Mean-variance optimized portfolio",
        "Highest expected return on the efficient frontier for a volatility target set by risk tolerance.",
    ),
}
_RANGE_ERRORS = {"greater_than_equal", "less_than_equal", "int_parsing", "float_parsing", "int_from_float",
                 "greater_than", "less_than", "finite_number"}


class PortfolioServiceError(Exception):
    """A failure the UI can show to the user."""


class InputValidationError(PortfolioServiceError):
    """Invalid form inputs; ``field_errors`` maps field name to a user-facing message."""

    def __init__(self, field_errors: dict[str, str]) -> None:
        self.field_errors = field_errors
        super().__init__("; ".join(field_errors.values()))


class PortfolioService:
    def __init__(
        self,
        engine: PortfolioOptimizationEngine | None = None,
        projection: ProjectionConfig | None = None,
    ) -> None:
        self._engine = engine
        self.projection = projection or ProjectionConfig()
        self._lock = threading.Lock()

    @property
    def engine(self) -> PortfolioOptimizationEngine:
        if self._engine is None:
            self._engine = get_optimization_engine()
        return self._engine

    @staticmethod
    def get_form_options() -> dict[str, dict[str, Any]]:
        return get_form_options()

    @staticmethod
    def validate(request: PortfolioRequest | Mapping[str, Any]) -> PortfolioRequest:
        if isinstance(request, PortfolioRequest):
            return request
        try:
            return PortfolioRequest.model_validate(dict(request))
        except ValidationError as exc:
            raise InputValidationError(_field_errors(exc)) from None

    def build_portfolio(self, request: PortfolioRequest | Mapping[str, Any]) -> PortfolioResponse:
        req = self.validate(request)
        risk = req.risk_score
        effective_risk, horizon_note = horizon_adjusted_risk(risk, req.horizon_years)
        profile = InvestorProfile(age=req.age, risk_tolerance=effective_risk)

        try:
            with self._lock:
                inputs = self.engine.market_inputs()
                rule = self.engine.optimize("rule_based", profile, inputs=inputs)
                mvo = self.engine.optimize("mean_variance", profile, inputs=inputs)
                names = self._fund_names([*rule.weights.index, *mvo.weights.index])
                data_as_of = self._data_as_of()
        except (OptimizationError, MarketDataError) as exc:
            raise PortfolioServiceError(f"Could not build a portfolio: {exc}") from exc

        if mvo.frontier is None:
            raise PortfolioServiceError("Mean-variance optimization did not return an efficient frontier")
        frontier_points = mvo.frontier.points

        summary = ProfileSummary(
            age=req.age,
            goal=req.goal,
            goal_label=req.goal.label,
            horizon_years=req.horizon_years,
            risk_tolerance_input=(req.risk_tolerance.label if isinstance(req.risk_tolerance, RiskLevel)
                                  else f"{req.risk_tolerance:g}"),
            risk_tolerance=risk,
            effective_risk_tolerance=effective_risk,
            risk_band=risk_band(effective_risk),
            horizon_adjustment=horizon_note,
        )
        rule_rec = self._recommendation("rule_based", rule, req, names, frontier_points)
        mvo_rec = self._recommendation("mean_variance", mvo, req, names, frontier_points)

        return PortfolioResponse(
            request=req,
            profile=summary,
            market_data=_market_summary(inputs, data_as_of),
            rule_based=rule_rec,
            mean_variance=mvo_rec,
            efficient_frontier=_frontier_data(mvo, rule_rec, mvo_rec),
            notes=build_notes(req, summary, rule_rec, mvo_rec),
            generated_at=datetime.now(timezone.utc).replace(microsecond=0),
        )

    # -- helpers -----------------------------------------------------------
    def _recommendation(
        self,
        method: str,
        result: AllocationResult,
        req: PortfolioRequest,
        names: dict[str, str],
        frontier_points: pd.DataFrame,
    ) -> PortfolioRecommendation:
        weights = result.weights[result.weights > 0].sort_values(ascending=False)
        amounts = _allocate_amounts(weights, req.initial_investment)
        holdings = []
        for ticker, weight in weights.items():
            asset_class = get_asset_class("cash") if ticker == CASH else get_asset_class_for_ticker(ticker)
            holdings.append(Holding(ticker=ticker, name=names.get(ticker, ticker), asset_class=asset_class.name,
                                    role=asset_class.role, weight=float(weight), amount=amounts[ticker]))

        classes: dict[str, AssetClassAllocation] = {}
        for h in holdings:
            entry = classes.setdefault(h.asset_class, AssetClassAllocation(asset_class=h.asset_class, role=h.role,
                                                                           weight=0.0, amount=0.0))
            entry.weight += h.weight
            entry.amount = round(entry.amount + h.amount, 2)

        m = result.metrics
        title, description = METHOD_TEXT[method]
        return PortfolioRecommendation(
            method=method,
            title=title,
            description=description,
            expected_return=m.expected_return,
            volatility=m.volatility,
            sharpe_ratio=m.sharpe_ratio,
            holdings=holdings,
            asset_classes=sorted(classes.values(), key=lambda a: -a.weight),
            frontier_position=locate_on_frontier(m.volatility, m.expected_return, frontier_points),
            projection=project_portfolio_value(req.initial_investment, req.monthly_contribution, req.horizon_years,
                                               m.expected_return, m.volatility, self.projection),
            details=_json_safe({k: v for k, v in result.details.items() if k != "estimation_window"}),
        )

    def _fund_names(self, tickers: list[str]) -> dict[str, str]:
        names = {CASH: get_asset_class("cash").name}
        for ticker in dict.fromkeys(tickers):
            if ticker == CASH:
                continue
            try:
                names[ticker] = self.engine.market_data.get_info(ticker).get("longName") or ticker
            except (MarketDataError, OSError, ValueError):
                names[ticker] = ticker
        return names

    def _data_as_of(self) -> str | None:
        last = self.engine.market_data.cache_status()["last_date"].dropna()
        return None if last.empty else str(max(last))


def _allocate_amounts(weights: pd.Series, total: float) -> dict[str, float]:
    """Dollar amounts rounded to cents that add up exactly to ``total``."""
    amounts = {t: round(float(w) * total, 2) for t, w in weights.items()}
    remainder = round(total - sum(amounts.values()), 2)
    if remainder and amounts:
        largest = max(amounts, key=amounts.get)
        amounts[largest] = round(amounts[largest] + remainder, 2)
    return amounts


def _frontier_data(mvo: AllocationResult, rule_rec: PortfolioRecommendation,
                   mvo_rec: PortfolioRecommendation) -> EfficientFrontierData:
    frontier = mvo.frontier
    points = [
        FrontierPoint(
            expected_return=float(row["expected_return"]),
            volatility=float(row["volatility"]),
            sharpe_ratio=float(row["sharpe_ratio"]),
            weights={t: float(w) for t, w in frontier.weights.loc[idx].items() if w > 0},
        )
        for idx, row in frontier.points.iterrows()
    ]
    references = [
        ReferencePortfolio(
            name=name,
            expected_return=ref.metrics.expected_return,
            volatility=ref.metrics.volatility,
            sharpe_ratio=ref.metrics.sharpe_ratio,
            weights={t: float(w) for t, w in ref.weights.items() if w > 0},
        )
        for name, ref in mvo.reference_portfolios.items()
    ]
    return EfficientFrontierData(
        points=points,
        reference_portfolios=references,
        client_point=PortfolioPoint(label=mvo_rec.title, expected_return=mvo_rec.expected_return,
                                    volatility=mvo_rec.volatility),
        rule_based_point=PortfolioPoint(label=rule_rec.title, expected_return=rule_rec.expected_return,
                                        volatility=rule_rec.volatility),
    )


def _market_summary(inputs: MarketInputs, data_as_of: str | None) -> MarketDataSummary:
    window = inputs.estimation_window
    return MarketDataSummary(
        estimation_start=window["start"],
        estimation_end=window["end"],
        observations=int(window["observations"]),
        frequency=str(window["frequency"]),
        risk_free_rate=inputs.risk_free_rate,
        data_as_of=data_as_of,
    )


def _field_errors(exc: ValidationError) -> dict[str, str]:
    options = get_form_options()
    errors: dict[str, str] = {}
    for err in exc.errors():
        field = str(err["loc"][0]) if err["loc"] else "request"
        if field in errors:
            continue
        opt = options.get(field)
        if opt is None:
            errors[field] = f"Unknown input '{field}'" if err["type"] == "extra_forbidden" else err["msg"]
        elif err["type"] in _RANGE_ERRORS and opt.get("minimum") is not None:
            errors[field] = f"{opt['label']} must be a number between {opt['minimum']:,} and {opt['maximum']:,}"
        elif "choices" in opt and err["type"] in {"enum", "value_error"}:
            labels = ", ".join(c["label"] for c in opt["choices"])
            prefix = "a number from 1 to 10 or one of: " if field == "risk_tolerance" else "one of: "
            errors[field] = f"{opt['label']} must be {prefix}{labels}"
        else:
            errors[field] = f"{opt['label']}: {err['msg']}"
    return errors


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


# -- module-level entry points for UIs ----------------------------------------------
_service: PortfolioService | None = None
_service_lock = threading.Lock()


def get_portfolio_service() -> PortfolioService:
    """Process-wide service; the optimization engine and market data are loaded once."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = PortfolioService()
    return _service


def recommend_portfolio(
    risk_tolerance: float | str,
    horizon_years: int,
    initial_investment: float,
    monthly_contribution: float,
    goal: str,
    age: int,
) -> PortfolioResponse:
    """Build both portfolios from UI form values (arguments in form order, e.g. Gradio inputs)."""
    return get_portfolio_service().build_portfolio(
        {
            "risk_tolerance": risk_tolerance,
            "horizon_years": horizon_years,
            "initial_investment": initial_investment,
            "monthly_contribution": monthly_contribution,
            "goal": goal,
            "age": age,
        }
    )
