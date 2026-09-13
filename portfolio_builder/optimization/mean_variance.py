"""Mean-variance optimization with scipy.optimize.minimize (SLSQP).

All portfolios are long-only (0 <= w_i <= max_weight) and fully invested (sum(w) == 1). An optional
``GroupBound`` keeps the total weight of a group of assets (e.g. equity funds) inside a band.
Solver functions work on plain numpy arrays of annualized expected returns ``mu`` and
covariance ``cov``; ``MeanVarianceStrategy`` wraps them for the engine.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from ..universe import equity_tickers
from .base import AllocationStrategy, OptimizationError
from .glide_path import DEFAULT_MAX_EQUITY, DEFAULT_MAX_RISK_SHIFT, DEFAULT_MIN_EQUITY, GlidePath, equity_target
from .inputs import MarketInputs, portfolio_metrics
from .models import AllocationResult, EfficientFrontier, InvestorProfile, Portfolio

logger = logging.getLogger(__name__)

CONSTRAINT_TOLERANCE = 1e-6
_SOLVER_OPTIONS = {"ftol": 1e-12, "maxiter": 1000}

OBJECTIVES = ("risk_tolerance", "max_sharpe", "min_volatility", "target_volatility", "target_return")


@dataclass(frozen=True, eq=False)
class GroupBound:
    """Keep ``lower <= sum(w[mask]) <= upper`` (e.g. equity funds within a glide-path band)."""

    mask: np.ndarray
    lower: float
    upper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask", np.asarray(self.mask, dtype=bool))
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise OptimizationError(f"Invalid group band [{self.lower}, {self.upper}]")

    def total(self, w: np.ndarray) -> float:
        return float(np.asarray(w)[self.mask].sum())

    def constraints(self) -> list[dict[str, Any]]:
        indicator = self.mask.astype(float)
        return [
            {"type": "ineq", "fun": lambda w: float(w @ indicator) - self.lower, "jac": lambda w: indicator},
            {"type": "ineq", "fun": lambda w: self.upper - float(w @ indicator), "jac": lambda w: -indicator},
        ]

    def check_feasible(self, n: int, max_weight: float) -> None:
        if self.mask.shape != (n,):
            raise OptimizationError(f"Group mask has shape {self.mask.shape}; expected ({n},)")
        k = int(self.mask.sum())
        most = min(1.0, k * max_weight)             # all weight the group can take
        least = max(0.0, 1.0 - (n - k) * max_weight)  # weight the group must take
        if self.upper < least - CONSTRAINT_TOLERANCE or self.lower > most + CONSTRAINT_TOLERANCE:
            raise OptimizationError(
                f"Group band [{self.lower:.1%}, {self.upper:.1%}] is infeasible: the group can hold "
                f"between {least:.1%} and {most:.1%} with max_weight={max_weight}"
            )

    def feasible_start(self, n: int, max_weight: float) -> np.ndarray:
        """Equal weights inside and outside the group, splitting the budget at the band midpoint."""
        k = int(self.mask.sum())
        least, most = max(0.0, 1.0 - (n - k) * max_weight), min(1.0, k * max_weight)
        share = min(max((self.lower + self.upper) / 2, least), most)
        w = np.zeros(n)
        if k:
            w[self.mask] = share / k
        if n - k:
            w[~self.mask] = (1.0 - share) / (n - k)
        return w


# -- solvers -------------------------------------------------------------------
def _check_problem(mu: np.ndarray, cov: np.ndarray | None, max_weight: float, group: GroupBound | None = None) -> int:
    n = len(mu)
    if n == 0:
        raise OptimizationError("No assets to optimize")
    if cov is not None and cov.shape != (n, n):
        raise OptimizationError(f"Covariance shape {cov.shape} does not match {n} assets")
    if not 0 < max_weight <= 1:
        raise OptimizationError("max_weight must be in (0, 1]")
    if max_weight * n < 1 - CONSTRAINT_TOLERANCE:
        raise OptimizationError(f"max_weight={max_weight} is infeasible for {n} assets (weights must sum to 1)")
    if group is not None:
        group.check_feasible(n, max_weight)
    return n


def _solve(
    objective: Callable[[np.ndarray], float],
    jac: Callable[[np.ndarray], np.ndarray],
    n: int,
    max_weight: float,
    x0: np.ndarray | None = None,
    constraints: Sequence[dict[str, Any]] = (),
    group: GroupBound | None = None,
) -> np.ndarray:
    budget = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}
    all_constraints = [*constraints, *(group.constraints() if group is not None else [])]
    if x0 is None:
        x0 = group.feasible_start(n, max_weight) if group is not None else np.full(n, 1.0 / n)
    result = minimize(
        objective,
        np.asarray(x0, dtype=float),
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[budget, *all_constraints],
        options=_SOLVER_OPTIONS,
    )
    return _finalize(result, max_weight, all_constraints)


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


def min_volatility(mu: np.ndarray, cov: np.ndarray, max_weight: float = 1.0,
                   group: GroupBound | None = None) -> np.ndarray:
    """Minimum-variance long-only portfolio."""
    n = _check_problem(mu, cov, max_weight, group)
    fun, jac = _variance(cov)
    return _solve(fun, jac, n, max_weight, group=group)


def max_return(mu: np.ndarray, max_weight: float = 1.0, group: GroupBound | None = None) -> np.ndarray:
    """Highest expected return.

    Without a group this fills assets in descending return order up to ``max_weight``; with a group it
    solves the linear program (HiGHS) under the bounds, budget and group band.
    """
    n = _check_problem(mu, None, max_weight, group)
    if group is None:
        w = np.zeros(n)
        remaining = 1.0
        for i in np.argsort(-mu, kind="stable"):
            w[i] = min(max_weight, remaining)
            remaining -= w[i]
            if remaining <= 1e-12:
                break
        return w
    indicator = group.mask.astype(float)
    result = linprog(
        -np.asarray(mu, dtype=float),
        A_ub=np.vstack([indicator, -indicator]),
        b_ub=np.array([group.upper, -group.lower]),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, max_weight)] * n,
        method="highs",
    )
    if not result.success:
        raise OptimizationError(f"Max-return linear program failed: {result.message}")
    w = np.clip(result.x, 0.0, max_weight)
    w[w < 1e-12] = 0.0
    return w / w.sum()


def efficient_return(mu: np.ndarray, cov: np.ndarray, target_return: float, max_weight: float = 1.0,
                     x0: np.ndarray | None = None, group: GroupBound | None = None) -> np.ndarray:
    """Minimum-variance portfolio with expected return equal to ``target_return``."""
    n = _check_problem(mu, cov, max_weight, group)
    lo, hi = float(mu @ max_return(-mu, max_weight, group)), float(mu @ max_return(mu, max_weight, group))
    if not lo - CONSTRAINT_TOLERANCE <= target_return <= hi + CONSTRAINT_TOLERANCE:
        raise OptimizationError(f"Target return {target_return:.4%} outside achievable range [{lo:.4%}, {hi:.4%}]")
    fun, jac = _variance(cov)
    target_con = {"type": "eq", "fun": lambda w: float(w @ mu) - target_return, "jac": lambda w: mu}
    return _solve(fun, jac, n, max_weight, x0=x0, constraints=[target_con], group=group)


def efficient_risk(mu: np.ndarray, cov: np.ndarray, target_volatility: float, max_weight: float = 1.0,
                   group: GroupBound | None = None) -> tuple[np.ndarray, str | None]:
    """Maximum-return portfolio with volatility at most ``target_volatility``.

    Returns ``(weights, note)``; ``note`` explains when the target was outside the achievable
    range and the min-volatility or max-return portfolio was used instead.
    """
    n = _check_problem(mu, cov, max_weight, group)
    w_min = min_volatility(mu, cov, max_weight, group)
    vol_min = float(np.sqrt(w_min @ cov @ w_min))
    if target_volatility <= vol_min:
        return w_min, f"target volatility {target_volatility:.4%} at or below minimum {vol_min:.4%}; using min-volatility portfolio"
    w_max = max_return(mu, max_weight, group)
    vol_max = float(np.sqrt(w_max @ cov @ w_max))
    if target_volatility >= vol_max:
        return w_max, f"target volatility {target_volatility:.4%} at or above max-return portfolio {vol_max:.4%}; using max-return portfolio"

    risk_con = {
        "type": "ineq",
        "fun": lambda w: target_volatility**2 - float(w @ cov @ w),
        "jac": lambda w: -2.0 * cov @ w,
    }
    w = _solve(lambda w: -float(w @ mu), lambda w: -mu, n, max_weight, x0=w_min, constraints=[risk_con], group=group)
    return w, None


def max_sharpe(mu: np.ndarray, cov: np.ndarray, risk_free_rate: float, max_weight: float = 1.0,
               starts: Sequence[np.ndarray] = (), group: GroupBound | None = None) -> np.ndarray:
    """Maximum Sharpe ratio portfolio, solved from several starting points (best result wins)."""
    n = _check_problem(mu, cov, max_weight, group)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(max(w @ cov @ w, 1e-16))
        return -(float(w @ mu) - risk_free_rate) / vol

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        var = max(w @ cov @ w, 1e-16)
        vol = np.sqrt(var)
        excess = float(w @ mu) - risk_free_rate
        return -(mu / vol - excess * (cov @ w) / (var * vol))

    first = group.feasible_start(n, max_weight) if group is not None else np.full(n, 1.0 / n)
    candidates = [first, min_volatility(mu, cov, max_weight, group), max_return(mu, max_weight, group), *starts]
    best: np.ndarray | None = None
    best_value = np.inf
    for x0 in candidates:
        try:
            w = _solve(neg_sharpe, neg_sharpe_grad, n, max_weight, x0=x0, group=group)
        except OptimizationError as exc:
            logger.debug("max_sharpe start failed: %s", exc)
            continue
        value = neg_sharpe(w)
        if value < best_value - 1e-12:
            best, best_value = w, value
    if best is None:
        raise OptimizationError("Max Sharpe optimization failed from every starting point")
    return best


def efficient_frontier(inputs: MarketInputs, n_points: int = 50, max_weight: float = 1.0,
                       group: GroupBound | None = None) -> EfficientFrontier:
    """Long-only efficient frontier from the min-volatility portfolio up to the max-return portfolio."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    mu, cov = inputs.expected_returns.to_numpy(), inputs.covariance.to_numpy()
    w_min = min_volatility(mu, cov, max_weight, group)
    w_max = max_return(mu, max_weight, group)
    r_min, r_max = float(w_min @ mu), float(w_max @ mu)

    solutions: list[np.ndarray] = [w_min]
    previous = w_min
    for target in np.linspace(r_min, r_max, n_points)[1:-1]:
        try:
            previous = efficient_return(mu, cov, float(target), max_weight, x0=previous, group=group)
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

    With a ``glide_path``, equity funds must total the glide-path equity share (after the same
    risk-tolerance shift and clamp as the rule-based strategy) within ``±equity_band``; the frontier,
    reference portfolios and client portfolio are all solved under that band.
    """

    name = "mean_variance"

    def __init__(
        self,
        objective: str = "risk_tolerance",
        target: float | None = None,
        n_points: int = 50,
        max_weight: float = 1.0,
        vol_range: tuple[float, float] = (0.0, 1.0),
        glide_path: GlidePath | None = None,
        equity_band: float = 0.05,
        max_risk_shift: float = DEFAULT_MAX_RISK_SHIFT,
        min_equity: float = DEFAULT_MIN_EQUITY,
        max_equity: float = DEFAULT_MAX_EQUITY,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        if objective in ("target_volatility", "target_return") and target is None:
            raise ValueError(f"objective {objective!r} requires a target")
        if not 0.0 <= vol_range[0] <= vol_range[1] <= 1.0:
            raise ValueError("vol_range must satisfy 0 <= low <= high <= 1")
        if not 0.0 <= equity_band <= 1.0:
            raise ValueError("equity_band must be between 0 and 1")
        self.objective = objective
        self.target = target
        self.n_points = n_points
        self.max_weight = max_weight
        self.vol_range = vol_range
        self.glide_path = glide_path
        self.equity_band = equity_band
        self.max_risk_shift = max_risk_shift
        self.min_equity = min_equity
        self.max_equity = max_equity

    def equity_group(self, profile: InvestorProfile, inputs: MarketInputs) -> tuple[GroupBound, float] | None:
        """Equity band for the profile (``None`` without a glide path)."""
        if self.glide_path is None:
            return None
        equity, _ = equity_target(self.glide_path, profile, self.max_risk_shift, self.min_equity, self.max_equity)
        mask = np.isin(inputs.tickers, equity_tickers())
        if not mask.any() or mask.all():
            raise OptimizationError("A glide-path equity band needs both equity and non-equity funds in the inputs")
        group = GroupBound(mask, max(0.0, equity - self.equity_band), min(1.0, equity + self.equity_band))
        return group, equity

    def allocate(self, profile: InvestorProfile, inputs: MarketInputs) -> AllocationResult:
        mu, cov = inputs.expected_returns.to_numpy(), inputs.covariance.to_numpy()
        banded = self.equity_group(profile, inputs)
        group = banded[0] if banded else None
        frontier = efficient_frontier(inputs, self.n_points, self.max_weight, group)
        best_point = frontier.points["sharpe_ratio"].idxmax()

        references = {
            "min_volatility": frontier.weights.iloc[0].to_numpy(),
            "max_sharpe": max_sharpe(
                mu, cov, inputs.risk_free_rate, self.max_weight,
                starts=[frontier.weights.loc[best_point].to_numpy()], group=group,
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
            weights, note = efficient_risk(mu, cov, target_vol, self.max_weight, group)
            details.update(target_volatility=target_vol, volatility_range=(vol_min, vol_max), note=note)
        elif self.objective == "target_volatility":
            weights, note = efficient_risk(mu, cov, float(self.target), self.max_weight, group)
            details.update(target_volatility=self.target, note=note)
        elif self.objective == "target_return":
            weights = efficient_return(mu, cov, float(self.target), self.max_weight, group=group)
            details.update(target_return=self.target)
        elif self.objective == "max_sharpe":
            weights = references["max_sharpe"]
        else:
            weights = references["min_volatility"]

        if banded is not None:
            group, equity = banded
            details.update(
                glide_path=self.glide_path.name,
                equity_target=equity,
                equity_band=(group.lower, group.upper),
                equity_weight=group.total(weights),
            )

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
