"""Project portfolio value over the investment horizon with monthly contributions.

Two methods share one return model: annual returns are lognormal with arithmetic mean ``expected_return``
and standard deviation ``volatility``.

* ``monte_carlo`` (default): simulate month-by-month paths; percentiles and goal odds come from the
  simulated balances, and a few rank-spaced sample paths are kept for charting.
* ``simple_percentiles``: no simulation. Each percentile line grows the initial amount and contributions at
  a constant annual return: the percentile of the horizon's annualized return distribution. Goal odds are
  the probability that the annualized return beats the constant return needed to reach the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .schemas import PROJECTION_LABELS, PROJECTION_METHODS, Projection, ProjectionMethod, ProjectionPoint

PERCENTILES = (10.0, 25.0, 50.0, 75.0, 90.0)
# Bracket for solving the constant annual return that reaches a target (-99% to +10,000% a year).
_RATE_BRACKET = (-0.99, 100.0)


@dataclass(frozen=True)
class ProjectionConfig:
    simulations: int = 5000
    seed: int | None = 42
    sample_paths: int = 30  # simulated paths kept for charting (Monte Carlo only)


def project_portfolio_value(
    initial: float,
    monthly_contribution: float,
    years: int,
    expected_return: float,
    volatility: float,
    config: ProjectionConfig | None = None,
    target_amount: float | None = None,
    method: ProjectionMethod = "monte_carlo",
) -> Projection:
    """Expected path plus 10/25/50/75/90th percentiles, reported at each year end.

    Each month the balance grows, then the contribution is added. When ``target_amount`` is given,
    ``probability_of_meeting_target`` is the chance of ending at or above it under the chosen ``method``.
    """
    config = config or ProjectionConfig()
    if years < 1:
        raise ValueError("years must be at least 1")
    if expected_return <= -1:
        raise ValueError("expected_return must be greater than -100%")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if method not in PROJECTION_LABELS:
        raise ValueError(f"method must be one of {', '.join(PROJECTION_METHODS)}")

    months = years * 12
    growth = (1.0 + expected_return) ** (1.0 / 12.0)
    expected = np.empty(months + 1)
    expected[0] = initial
    for m in range(1, months + 1):
        expected[m] = expected[m - 1] * growth + monthly_contribution

    sigma_log = math.sqrt(math.log1p(volatility**2 / (1.0 + expected_return) ** 2))
    mu_log = math.log1p(expected_return) - sigma_log**2 / 2.0

    if method == "monte_carlo":
        bands_by_year, probability, sample_paths = _simulate(
            initial, monthly_contribution, years, expected, volatility, mu_log, sigma_log, config, target_amount)
        simulations, seed = config.simulations, config.seed
    else:
        bands_by_year, probability = _percentile_formula(
            initial, monthly_contribution, years, expected, volatility, mu_log, sigma_log, target_amount)
        sample_paths, simulations, seed = [], 0, None

    points = [ProjectionPoint(year=0, total_contributed=initial, expected=initial,
                              p10=initial, p25=initial, p50=initial, p75=initial, p90=initial)]
    for year in range(1, years + 1):
        p10, p25, p50, p75, p90 = (round(float(b), 2) for b in bands_by_year[year])
        points.append(
            ProjectionPoint(
                year=year,
                total_contributed=round(initial + monthly_contribution * 12 * year, 2),
                expected=round(float(expected[year * 12]), 2),
                p10=p10, p25=p25, p50=p50, p75=p75, p90=p90,
            )
        )

    final = points[-1]
    return Projection(
        method=method,
        points=points,
        final_expected=final.expected,
        final_p10=final.p10,
        final_p25=final.p25,
        final_p50=final.p50,
        final_p75=final.p75,
        final_p90=final.p90,
        total_contributed=final.total_contributed,
        target_amount=target_amount,
        probability_of_meeting_target=probability,
        simulations=simulations,
        seed=seed,
        sample_paths=sample_paths,
    )


def _simulate(
    initial: float,
    monthly_contribution: float,
    years: int,
    expected: np.ndarray,
    volatility: float,
    mu_log: float,
    sigma_log: float,
    config: ProjectionConfig,
    target_amount: float | None,
) -> tuple[dict[int, list[float]], float | None, list[list[float]]]:
    """Monte Carlo: percentile bands per year, goal odds, and rank-spaced sample paths (year-end values)."""
    months = years * 12
    rng = np.random.default_rng(config.seed)
    monthly_growth = np.exp(rng.normal(mu_log / 12.0, sigma_log / np.sqrt(12.0), size=(config.simulations, months)))

    balances = np.full(config.simulations, float(initial))
    yearly = np.empty((config.simulations, years + 1))
    yearly[:, 0] = initial
    bands: dict[int, list[float]] = {}
    for m in range(1, months + 1):
        balances = balances * monthly_growth[:, m - 1] + monthly_contribution
        if m % 12 == 0:
            year = m // 12
            yearly[:, year] = balances
            if volatility == 0:
                bands[year] = [expected[m]] * len(PERCENTILES)
            else:
                bands[year] = list(np.percentile(balances, PERCENTILES))

    probability = None
    if target_amount is not None:
        final_values = np.full(config.simulations, expected[-1]) if volatility == 0 else balances
        probability = float(np.mean(final_values >= target_amount))

    count = min(config.sample_paths, config.simulations)
    sample_paths: list[list[float]] = []
    if count > 0:
        by_final = np.argsort(yearly[:, -1], kind="stable")
        picks = by_final[np.linspace(0, config.simulations - 1, count).round().astype(int)]
        sample_paths = [[round(float(v), 2) for v in yearly[i]] for i in picks]
    return bands, probability, sample_paths


def _future_value(initial: float, monthly_contribution: float, annual_rate: float, months: int) -> float:
    """Balance after ``months`` at a constant annual rate: grow each month, then add the contribution."""
    g = (1.0 + annual_rate) ** (1.0 / 12.0)
    if abs(g - 1.0) < 1e-12:
        return initial + monthly_contribution * months
    return initial * g**months + monthly_contribution * (g**months - 1.0) / (g - 1.0)


def _percentile_formula(
    initial: float,
    monthly_contribution: float,
    years: int,
    expected: np.ndarray,
    volatility: float,
    mu_log: float,
    sigma_log: float,
    target_amount: float | None,
) -> tuple[dict[int, list[float]], float | None]:
    """Simple percentiles: constant percentile returns of the annualized return distribution for each horizon."""
    z_scores = [float(norm.ppf(p / 100.0)) for p in PERCENTILES]
    bands: dict[int, list[float]] = {}
    for year in range(1, years + 1):
        if volatility == 0:
            bands[year] = [expected[year * 12]] * len(PERCENTILES)
            continue
        rates = [math.exp(mu_log + z * sigma_log / math.sqrt(year)) - 1.0 for z in z_scores]
        bands[year] = [_future_value(initial, monthly_contribution, r, year * 12) for r in rates]

    probability = None
    if target_amount is not None:
        probability = _formula_goal_probability(initial, monthly_contribution, years, expected, volatility,
                                                mu_log, sigma_log, target_amount)
    return bands, probability


def _formula_goal_probability(
    initial: float,
    monthly_contribution: float,
    years: int,
    expected: np.ndarray,
    volatility: float,
    mu_log: float,
    sigma_log: float,
    target_amount: float,
) -> float:
    """Chance that the horizon's annualized return beats the constant return needed to reach the target."""
    if volatility == 0:
        return float(expected[-1] >= target_amount)
    months = years * 12
    low, high = _RATE_BRACKET
    if _future_value(initial, monthly_contribution, low, months) >= target_amount:
        return 1.0
    if _future_value(initial, monthly_contribution, high, months) < target_amount:
        return 0.0
    needed = brentq(lambda r: _future_value(initial, monthly_contribution, r, months) - target_amount, low, high,
                    xtol=1e-12)
    z = (math.log1p(needed) - mu_log) / (sigma_log / math.sqrt(years))
    return float(norm.sf(z))
