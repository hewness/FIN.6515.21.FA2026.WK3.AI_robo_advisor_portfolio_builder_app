"""Mean-variance optimization with scipy.optimize.minimize (SLSQP).

All portfolios are long-only (0 <= w_i <= max_weight) and fully invested (sum(w) == 1).
Solver functions work on plain numpy arrays of annualized expected returns ``mu`` and
covariance ``cov``; ``MeanVarianceStrategy`` wraps them for the engine.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import AllocationStrategy, OptimizationError
from .inputs import MarketInputs, portfolio_metrics
from .models import AllocationResult, EfficientFrontier, InvestorProfile, Portfolio

logger = logging.getLogger(__name__)

CONSTRAINT_TOLERANCE = 1e-6
_SOLVER_OPTIONS = {"ftol": 1e-12, "maxiter": 1000}

OBJECTIVES = ("risk_tolerance", "max_sharpe", "min_volatility", "target_volatility", "target_return")


# -- solvers -------------------------------------------------------------------
def _check_problem(mu: np.ndarray, cov: np.ndarray | None, max_weight: float) -> int:
    n = len(mu)
    if n == 0:
        raise OptimizationError("No assets to optimize")
    if cov is not None and cov.shape != (n, n):
        raise OptimizationError(f"Covariance shape {cov.shape} does not match {n} assets")
    if not 0 < max_weight <= 1:
        raise OptimizationError("max_weight must be in (0, 1]")
    if max_weight * n < 1 - CONSTRAINT_TOLERANCE:
        raise OptimizationError(f"max_weight={max_weight} is infeasible for {n} assets (weights must sum to 1)")
    return n


def _solve(
    objective: Callable[[np.ndarray], float],
    jac: Callable[[np.ndarray], np.ndarray],
    n: int,
    max_weight: float,
    x0: np.ndarray | None = None,
    constraints: Sequence[dict[str, Any]] = (),
) -> np.ndarray:
    budget = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}
    result = minimize(
        objective,
        np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, dtype=float),
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[budget, *constraints],
        options=_SOLVER_OPTIONS,
    )
    return _finalize(result, max_weight, constraints)


def _finalize(result: Any, max_weight: float, constraints: Sequence[dict[str, Any]]) -> np.ndarray:
    w = np.asarray(result.x, dtype=float)
    if not result.success:
        raise OptimizationError(f"Optimizer did not converge: {result.message}")
    violations = [
        w.min() < -CONSTRAINT_TOLERANCE,
        w.max() > max_weight + CONSTRAINT_TOLERANCE,
        abs(w.sum() - 1.0) > CONSTRAINT_TOLERANCE,
    ]
    for con in constraints:
        value = con["fun"](w)
        violations.append(abs(value) > CONSTRAINT_TOLERANCE if con["type"] == "eq" else value < -CONSTRAINT_TOLERANCE)
    if any(violations):
        raise OptimizationError("Optimizer result violates portfolio constraints")
    w = np.clip(w, 0.0, max_weight)
    w[w < 1e-10] = 0.0
    return w / w.sum()


def _variance(cov: np.ndarray) -> tuple[Callable, Callable]:
    return (lambda w: float(w @ cov @ w)), (lambda w: 2.0 * cov @ w)


def min_volatility(mu: np.ndarray, cov: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    """Minimum-variance long-only portfolio."""
    n = _check_problem(mu, cov, max_weight)
    fun, jac = _variance(cov)
    return _solve(fun, jac, n, max_weight)


def max_return(mu: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    """Highest expected return: fill assets in descending return order up to max_weight."""
    n = _check_problem(mu, None, max_weight)
    w = np.zeros(n)
    remaining = 1.0
    for i in np.argsort(-mu, kind="stable"):
        w[i] = min(max_weight, remaining)
        remaining -= w[i]
        if remaining <= 1e-12:
            break
    return w


def efficient_return(mu: np.ndarray, cov: np.ndarray, target_return: float, max_weight: float = 1.0,
                     x0: np.ndarray | None = None) -> np.ndarray:
    """Minimum-variance portfolio with expected return equal to ``target_return``."""
    n = _check_problem(mu, cov, max_weight)
    lo, hi = float(mu @ max_return(-mu, max_weight)), float(mu @ max_return(mu, max_weight))
    if not lo - CONSTRAINT_TOLERANCE <= target_return <= hi + CONSTRAINT_TOLERANCE:
        raise OptimizationError(f"Target return {target_return:.4%} outside achievable range [{lo:.4%}, {hi:.4%}]")
    fun, jac = _variance(cov)
    target_con = {"type": "eq", "fun": lambda w: float(w @ mu) - target_return, "jac": lambda w: mu}
    return _solve(fun, jac, n, max_weight, x0=x0, constraints=[target_con])


def efficient_risk(mu: np.ndarray, cov: np.ndarray, target_volatility: float, max_weight: float = 1.0,
                   ) -> tuple[np.ndarray, str | None]:
    """Maximum-return portfolio with volatility at most ``target_volatility``.

    Returns ``(weights, note)``; ``note`` explains when the target was outside the achievable
    range and the min-volatility or max-return portfolio was used instead.
    """
    n = _check_problem(mu, cov, max_weight)
    w_min = min_volatility(mu, cov, max_weight)
    vol_min = float(np.sqrt(w_min @ cov @ w_min))
    if target_volatility <= vol_min:
        return w_min, f"target volatility {target_volatility:.4%} at or below minimum {vol_min:.4%}; using min-volatility portfolio"
    w_max = max_return(mu, max_weight)
    vol_max = float(np.sqrt(w_max @ cov @ w_max))
    if target_volatility >= vol_max:
        return w_max, f"target volatility {target_volatility:.4%} at or above max-return portfolio {vol_max:.4%}; using max-return portfolio"

    risk_con = {
        "type": "ineq",
        "fun": lambda w: target_volatility**2 - float(w @ cov @ w),
        "jac": lambda w: -2.0 * cov @ w,
    }
    w = _solve(lambda w: -float(w @ mu), lambda w: -mu, n, max_weight, x0=w_min, constraints=[risk_con])
    return w, None


def max_sharpe(mu: np.ndarray, cov: np.ndarray, risk_free_rate: float, max_weight: float = 1.0,
               starts: Sequence[np.ndarray] = ()) -> np.ndarray:
    """Maximum Sharpe ratio portfolio, solved from several starting points (best result wins)."""
    n = _check_problem(mu, cov, max_weight)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(max(w @ cov @ w, 1e-16))
        return -(float(w @ mu) - risk_free_rate) / vol

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        var = max(w @ cov @ w, 1e-16)
        vol = np.sqrt(var)
        excess = float(w @ mu) - risk_free_rate
        return -(mu / vol - excess * (cov @ w) / (var * vol))

    candidates = [np.full(n, 1.0 / n), min_volatility(mu, cov, max_weight), max_return(mu, max_weight), *starts]
    best: np.ndarray | None = None
    best_value = np.inf
    for x0 in candidates:
        try:
            w = _solve(neg_sharpe, neg_sharpe_grad, n, max_weight, x0=x0)
        except OptimizationError as exc:
            logger.debug("max_sharpe start failed: %s", exc)
            continue
        value = neg_sharpe(w)
        if value < best_value - 1e-12:
            best, best_value = w, value
    if best is None:
        raise OptimizationError("Max Sharpe optimization failed from every starting point")
    return best


def efficient_frontier(inputs: MarketInputs, n_points: int = 50, max_weight: float = 1.0) -> EfficientFrontier:
    """Long-only efficient frontier from the min-volatility portfolio up to the max-return portfolio."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    mu, cov = inputs.expected_returns.to_numpy(), inputs.covariance.to_numpy()
    w_min = min_volatility(mu, cov, max_weight)
    w_max = max_return(mu, max_weight)
    r_min, r_max = float(w_min @ mu), float(w_max @ mu)

    solutions: list[np.ndarray] = [w_min]
    previous = w_min
    for target in np.linspace(r_min, r_max, n_points)[1:-1]:
        try:
            previous = efficient_return(mu, cov, float(target), max_weight, x0=previous)
            solutions.append(previous)
        except OptimizationError as exc:
            logger.warning("Skipping frontier point at return %.4f%%: %s", target * 100, exc)
    if r_max - r_min > CONSTRAINT_TOLERANCE:
        solutions.append(w_max)

    weights = pd.DataFrame(solutions, columns=inputs.tickers)
    weights.index.name = "point"
    points = pd.DataFrame(
        [portfolio_metrics(weights.loc[i], inputs).as_dict() for i in weights.index], index=weights.index
    )
    return EfficientFrontier(points=points, weights=weights)


# -- strategy ------------------------------------------------------------------
class MeanVarianceStrategy(AllocationStrategy):
    """Mean-variance optimizer returning the efficient frontier and the client's portfolio.

    Objectives:
      - ``risk_tolerance`` (default): target volatility interpolated between the min-volatility
        and max-return portfolios by the client's risk tolerance (scaled into ``vol_range``).
      - ``max_sharpe``, ``min_volatility``
      - ``target_volatility`` / ``target_return``: require ``target`` (annualized, e.g. 0.10).
    """

    name = "mean_variance"

    def __init__(
        self,
        objective: str = "risk_tolerance",
        target: float | None = None,
        n_points: int = 50,
        max_weight: float = 1.0,
        vol_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        if objective in ("target_volatility", "target_return") and target is None:
            raise ValueError(f"objective {objective!r} requires a target")
        if not 0.0 <= vol_range[0] <= vol_range[1] <= 1.0:
            raise ValueError("vol_range must satisfy 0 <= low <= high <= 1")
        self.objective = objective
        self.target = target
        self.n_points = n_points
        self.max_weight = max_weight
        self.vol_range = vol_range

    def allocate(self, profile: InvestorProfile, inputs: MarketInputs) -> AllocationResult:
        mu, cov = inputs.expected_returns.to_numpy(), inputs.covariance.to_numpy()
        frontier = efficient_frontier(inputs, self.n_points, self.max_weight)
        best_point = frontier.points["sharpe_ratio"].idxmax()

        references = {
            "min_volatility": frontier.weights.iloc[0].to_numpy(),
            "max_sharpe": max_sharpe(
                mu, cov, inputs.risk_free_rate, self.max_weight, starts=[frontier.weights.loc[best_point].to_numpy()]
            ),
            "max_return": frontier.weights.iloc[-1].to_numpy(),
        }
        vol_min = _vol(references["min_volatility"], cov)
        vol_max = _vol(references["max_return"], cov)

        details: dict[str, Any] = {
            "objective": self.objective,
            "estimation_window": inputs.estimation_window,
            "risk_free_rate": inputs.risk_free_rate,
            "max_weight": self.max_weight,
        }
        if self.objective == "risk_tolerance":
            lo, hi = self.vol_range
            fraction = lo + profile.risk_fraction * (hi - lo)
            target_vol = vol_min + fraction * (vol_max - vol_min)
            weights, note = efficient_risk(mu, cov, target_vol, self.max_weight)
            details.update(target_volatility=target_vol, volatility_range=(vol_min, vol_max), note=note)
        elif self.objective == "target_volatility":
            weights, note = efficient_risk(mu, cov, float(self.target), self.max_weight)
            details.update(target_volatility=self.target, note=note)
        elif self.objective == "target_return":
            weights = efficient_return(mu, cov, float(self.target), self.max_weight)
            details.update(target_return=self.target)
        elif self.objective == "max_sharpe":
            weights = references["max_sharpe"]
        else:
            weights = references["min_volatility"]

        series = pd.Series(weights, index=inputs.tickers)
        return AllocationResult(
            method=self.name,
            profile=profile,
            weights=series,
            metrics=portfolio_metrics(series, inputs),
            frontier=frontier,
            reference_portfolios={
                name: Portfolio(pd.Series(w, index=inputs.tickers),
                                portfolio_metrics(pd.Series(w, index=inputs.tickers), inputs))
                for name, w in references.items()
            },
            details=details,
        )


def _vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(w @ cov @ w, 0.0)))
