"""Research-informed allocation: equity share from wealth and income, not just age.

Implements the approximately optimal equity share from Choi, Liu and Liu (2025), "Practical Finance:
An Approximate Solution to Lifecycle Portfolio Choice" (equation 9):

    equity = clip(merton_share * (1 + human_capital / financial_wealth), 0, 1)

* ``merton_share`` (equation 8) is the Merton (1969) share for an investor with no labor income,
  ``(log equity premium + sigma^2 / 2) / (gamma * sigma^2)``.
* ``human_capital`` is the present value of expected labor income until retirement plus retirement
  income (Social Security and pensions) to age 100. Human capital is bond-like, so an investor with
  a lot of it relative to savings can hold more stock in the financial portfolio.
* Income is discounted with the paper's age-varying one-year-ahead discount rates, approximated by
  the regressions in the third columns of Table 1 (working life) and Table 2 (retirement).

Risk tolerance (1-10) maps linearly onto the paper's relative risk aversion grid (10 down to 4).
Research findings from Duarte, Fonseca, Goodman and Parker (2022) motivate keeping retirees' equity
high: retirees still have long horizons and pension-like income.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from .base import AllocationStrategy, OptimizationError
from .inputs import MarketInputs, portfolio_metrics
from .models import DEFAULT_REPLACEMENT_RATE, MAX_RISK, MIN_RISK, AllocationResult, InvestorProfile
from .rule_based import sleeve_weights, validate_sleeves

# Table 1, column 3: one-year-ahead labor income discount rate during working life.
WORKING_LIFE_COEFFICIENTS = {
    "risk_aversion_div10": 0.087,
    "log_equity_premium": -0.267,
    "log_risk_free": 1.132,
    "permanent_shock_variance": 4.332,
    "transitory_shock_variance": 0.028,
    "replacement_rate": 0.010,
    "age_div100": -0.149,
    "age_div100_squared": 0.142,
    "constant": -0.020,
}
# Table 2, column 3: one-year-ahead discount rate for retirement income.
RETIREMENT_COEFFICIENTS = {
    "risk_aversion_div10": 0.0003,
    "log_equity_premium": -0.217,
    "log_risk_free": 0.893,
    "age_div100": 0.476,
    "age_div100_squared": -0.295,
    "constant": -0.166,
}


@dataclass(frozen=True)
class PracticalFinanceAssumptions:
    """Capital market and income-risk assumptions (Practical Finance baseline values)."""

    log_risk_free: float = 0.02  # real
    log_equity_premium: float = 0.04
    equity_volatility: float = 0.185
    permanent_income_sd: float = 0.130  # college graduate (paper's worked example)
    transitory_income_sd: float = 0.242
    end_age: int = 100
    # Relative risk aversion at risk tolerance 1 and 10 (the paper's grid spans 4-10).
    gamma_bounds: tuple[float, float] = (10.0, 4.0)

    def __post_init__(self) -> None:
        if self.equity_volatility <= 0:
            raise ValueError("equity_volatility must be positive")
        if min(self.gamma_bounds) <= 0:
            raise ValueError("gamma_bounds must be positive")


DEFAULT_ASSUMPTIONS = PracticalFinanceAssumptions()


@dataclass(frozen=True)
class ResearchEquity:
    """The pieces of the practical finance equity share."""

    equity: float
    merton_share: float
    risk_aversion: float
    human_capital: float
    financial_wealth: float
    replacement_rate: float
    first_discount_rate: float | None

    @property
    def human_capital_ratio(self) -> float:
        return self.human_capital / self.financial_wealth

    @property
    def unclipped_equity(self) -> float:
        return self.merton_share * (1.0 + self.human_capital_ratio)


def risk_aversion(risk_tolerance: float, a: PracticalFinanceAssumptions = DEFAULT_ASSUMPTIONS) -> float:
    """Relative risk aversion for a 1-10 risk tolerance (linear between ``gamma_bounds``)."""
    frac = (min(max(risk_tolerance, MIN_RISK), MAX_RISK) - MIN_RISK) / (MAX_RISK - MIN_RISK)
    high, low = a.gamma_bounds
    return high + (low - high) * frac


def merton_share(gamma: float, a: PracticalFinanceAssumptions = DEFAULT_ASSUMPTIONS) -> float:
    """Equation 8: optimal equity share with no labor income, clipped to [0, 1]."""
    var = a.equity_volatility ** 2
    return min(max((a.log_equity_premium + var / 2) / (gamma * var), 0.0), 1.0)


def working_discount_rate(age: float, gamma: float, replacement_rate: float,
                          a: PracticalFinanceAssumptions = DEFAULT_ASSUMPTIONS) -> float:
    """Discount rate applied at ``age`` to labor income arriving at ``age + 1`` (Table 1, column 3)."""
    c, x = WORKING_LIFE_COEFFICIENTS, age / 100
    return (c["risk_aversion_div10"] * gamma / 10 + c["log_equity_premium"] * a.log_equity_premium
            + c["log_risk_free"] * a.log_risk_free + c["permanent_shock_variance"] * a.permanent_income_sd ** 2
            + c["transitory_shock_variance"] * a.transitory_income_sd ** 2 + c["replacement_rate"] * replacement_rate
            + c["age_div100"] * x + c["age_div100_squared"] * x * x + c["constant"])


def retirement_discount_rate(age: float, gamma: float, a: PracticalFinanceAssumptions = DEFAULT_ASSUMPTIONS) -> float:
    """Discount rate applied at ``age`` to retirement income arriving at ``age + 1`` (Table 2, column 3)."""
    c, x = RETIREMENT_COEFFICIENTS, age / 100
    return (c["risk_aversion_div10"] * gamma / 10 + c["log_equity_premium"] * a.log_equity_premium
            + c["log_risk_free"] * a.log_risk_free + c["age_div100"] * x + c["age_div100_squared"] * x * x
            + c["constant"])


def replacement_rate(annual_income: float, retirement_income: float) -> float:
    if annual_income <= 0:
        return DEFAULT_REPLACEMENT_RATE
    return min(max(retirement_income / annual_income, 0.0), 1.0)


def human_capital_schedule(age: int, annual_income: float, retirement_income: float, retirement_age: int,
                           gamma: float, a: PracticalFinanceAssumptions = DEFAULT_ASSUMPTIONS) -> pd.DataFrame:
    """Table 3 layout: expected income, gross and cumulative discount rates and present value by age."""
    d = replacement_rate(annual_income, retirement_income)
    rows, cumulative = [], 1.0
    for t in range(age + 1, a.end_age + 1):
        working = t < retirement_age
        rate = working_discount_rate(t - 1, gamma, d, a) if working else retirement_discount_rate(t - 1, gamma, a)
        cumulative *= 1.0 + rate
        income = annual_income if working else retirement_income
        rows.append((t, income, 1.0 + rate, cumulative, income / cumulative))
    return pd.DataFrame(rows, columns=["age", "expected_income", "gross_discount_rate", "cumulative_discount_rate",
                                       "discounted_income"]).set_index("age")


def human_capital(age: int, annual_income: float, retirement_income: float, retirement_age: int, gamma: float,
                  a: PracticalFinanceAssumptions = DEFAULT_ASSUMPTIONS) -> float:
    """Present value of future labor and retirement income through ``end_age``."""
    schedule = human_capital_schedule(age, annual_income, retirement_income, retirement_age, gamma, a)
    return float(schedule["discounted_income"].sum())


def research_equity_share(profile: InvestorProfile,
                          a: PracticalFinanceAssumptions = DEFAULT_ASSUMPTIONS) -> ResearchEquity:
    """Equation 9: ``clip(merton_share * (1 + human_capital / financial_wealth), 0, 1)``."""
    if profile.financial_wealth is None:
        raise OptimizationError("The research-informed model needs financial wealth (the amount invested)")
    gamma = risk_aversion(profile.risk_tolerance, a)
    alpha = merton_share(gamma, a)
    benefit = profile.resolved_retirement_income
    schedule = human_capital_schedule(profile.age, profile.annual_income, benefit, profile.retirement_age, gamma, a)
    h = float(schedule["discounted_income"].sum())
    equity = min(max(alpha * (1.0 + h / profile.financial_wealth), 0.0), 1.0)
    return ResearchEquity(
        equity=equity,
        merton_share=alpha,
        risk_aversion=gamma,
        human_capital=h,
        financial_wealth=float(profile.financial_wealth),
        replacement_rate=replacement_rate(profile.annual_income, benefit),
        first_discount_rate=float(schedule["gross_discount_rate"].iloc[0] - 1.0) if len(schedule) else None,
    )


@dataclass(frozen=True)
class ResearchInformedConfig:
    assumptions: PracticalFinanceAssumptions = DEFAULT_ASSUMPTIONS
    # Equity leans toward global market-cap weights (about half non-US), per the academic benchmark
    # in Choi (2022); popular books hold far less abroad.
    equity_sleeve: dict[str, float] = field(
        default_factory=lambda: {
            "us_large_cap": 0.45,
            "intl_developed": 0.35,
            "emerging_markets": 0.12,
            "real_estate": 0.08,
        }
    )
    # Human capital already acts like a bond; the financial defensive sleeve is mostly bonds and TIPS.
    defensive_sleeve: dict[str, float] = field(
        default_factory=lambda: {"us_aggregate_bonds": 0.65, "tips": 0.30, "cash": 0.05}
    )

    def __post_init__(self) -> None:
        validate_sleeves(equity_sleeve=self.equity_sleeve, defensive_sleeve=self.defensive_sleeve)


class ResearchInformedStrategy(AllocationStrategy):
    name = "research_informed"

    def __init__(self, config: ResearchInformedConfig | None = None) -> None:
        self.config = config or ResearchInformedConfig()

    def equity_share(self, profile: InvestorProfile) -> ResearchEquity:
        return research_equity_share(profile, self.config.assumptions)

    def allocate(self, profile: InvestorProfile, inputs: MarketInputs) -> AllocationResult:
        result = self.equity_share(profile)
        class_weights, weights, ticker_order = sleeve_weights(
            result.equity, self.config.equity_sleeve, self.config.defensive_sleeve, profile.risk_fraction, inputs
        )
        series = pd.Series(weights, dtype="float64")
        return AllocationResult(
            method=self.name,
            profile=profile,
            weights=series,
            metrics=portfolio_metrics(series, inputs),
            details={
                "equity_pct": result.equity,
                "unclipped_equity_pct": result.unclipped_equity,
                "merton_share": result.merton_share,
                "risk_aversion": result.risk_aversion,
                "human_capital": result.human_capital,
                "financial_wealth": result.financial_wealth,
                "human_capital_ratio": result.human_capital_ratio,
                "replacement_rate": result.replacement_rate,
                "first_discount_rate": result.first_discount_rate,
                "annual_income": profile.annual_income,
                "retirement_income": profile.resolved_retirement_income,
                "retirement_age": profile.retirement_age,
                "assumptions": asdict(self.config.assumptions),
                "asset_class_weights": class_weights,
                "ticker_order_by_volatility": ticker_order,
                "estimation_window": inputs.estimation_window,
                "risk_free_rate": inputs.risk_free_rate,
            },
        )
