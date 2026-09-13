"""Project portfolio value over the investment horizon with monthly contributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import Projection, ProjectionPoint


@dataclass(frozen=True)
class ProjectionConfig:
    simulations: int = 5000
    seed: int | None = 42
    percentiles: tuple[float, float, float] = (10.0, 50.0, 90.0)


def project_portfolio_value(
    initial: float,
    monthly_contribution: float,
    years: int,
    expected_return: float,
    volatility: float,
    config: ProjectionConfig | None = None,
) -> Projection:
    """Expected path plus simulated percentile bands, reported at each year end.

    Each month the balance grows, then the contribution is added. Simulated annual returns
    are lognormal with arithmetic mean ``expected_return`` and standard deviation
    ``volatility``, so the simulated mean matches the expected path.
    """
    config = config or ProjectionConfig()
    if years < 1:
        raise ValueError("years must be at least 1")
    if expected_return <= -1:
        raise ValueError("expected_return must be greater than -100%")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    months = years * 12
    growth = (1.0 + expected_return) ** (1.0 / 12.0)

    expected = np.empty(months + 1)
    expected[0] = initial
    for m in range(1, months + 1):
        expected[m] = expected[m - 1] * growth + monthly_contribution

    sigma_log = np.sqrt(np.log1p(volatility**2 / (1.0 + expected_return) ** 2))
    mu_log = np.log1p(expected_return) - sigma_log**2 / 2.0
    rng = np.random.default_rng(config.seed)
    monthly_growth = np.exp(rng.normal(mu_log / 12.0, sigma_log / np.sqrt(12.0), size=(config.simulations, months)))

    low_q, mid_q, high_q = config.percentiles
    balances = np.full(config.simulations, float(initial))
    points = [ProjectionPoint(year=0, total_contributed=initial, expected=initial, p10=initial, p50=initial, p90=initial)]
    for m in range(1, months + 1):
        balances = balances * monthly_growth[:, m - 1] + monthly_contribution
        if m % 12 == 0:
            year = m // 12
            if volatility == 0:
                low = mid = high = expected[m]
            else:
                low, mid, high = np.percentile(balances, [low_q, mid_q, high_q])
            points.append(
                ProjectionPoint(
                    year=year,
                    total_contributed=round(initial + monthly_contribution * 12 * year, 2),
                    expected=round(float(expected[m]), 2),
                    p10=round(float(low), 2),
                    p50=round(float(mid), 2),
                    p90=round(float(high), 2),
                )
            )

    final = points[-1]
    return Projection(
        points=points,
        final_expected=final.expected,
        final_p10=final.p10,
        final_p50=final.p50,
        final_p90=final.p90,
        total_contributed=final.total_contributed,
        simulations=config.simulations,
        seed=config.seed,
    )
