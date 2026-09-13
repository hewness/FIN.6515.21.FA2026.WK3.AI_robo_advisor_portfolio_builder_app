"""Turn UI risk inputs into the engine's 1-10 risk tolerance, adjusted for investment horizon."""

from __future__ import annotations

from enum import Enum

MIN_RISK, MAX_RISK = 1.0, 10.0


class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

    @property
    def label(self) -> str:
        return self.value.capitalize()

    @property
    def score(self) -> float:
        return RISK_LABELS[self]


RISK_LABELS: dict[RiskLevel, float] = {
    RiskLevel.CONSERVATIVE: 3.0,
    RiskLevel.MODERATE: 5.5,
    RiskLevel.AGGRESSIVE: 8.0,
}

# (minimum horizon in years, risk shift). The first matching row from the top wins.
# Horizons under 3 years are capped at SHORT_HORIZON_RISK_CAP instead of shifted.
SHORT_HORIZON_YEARS = 3
SHORT_HORIZON_RISK_CAP = 3.0
HORIZON_RISK_SHIFTS: tuple[tuple[int, float], ...] = ((20, 1.0), (10, 0.0), (5, -1.0), (3, -2.0))

# Upper bound (inclusive) of effective risk for each band.
RISK_BANDS: tuple[tuple[float, str], ...] = ((3.5, "Conservative"), (7.0, "Moderate"), (MAX_RISK, "Aggressive"))


def resolve_risk_tolerance(value: float | str | RiskLevel) -> float:
    """A 1-10 number or a risk label (Conservative / Moderate / Aggressive) as a 1-10 number."""
    if isinstance(value, RiskLevel):
        return value.score
    if isinstance(value, str):
        text = value.strip().lower()
        try:
            return RiskLevel(text).score
        except ValueError:
            value = float(text)  # raises ValueError for unknown labels
    if isinstance(value, bool):
        raise ValueError("Risk tolerance must be a number or a label")
    risk = float(value)
    if not MIN_RISK <= risk <= MAX_RISK:
        raise ValueError(f"Risk tolerance must be between {MIN_RISK:g} and {MAX_RISK:g}")
    return risk


def horizon_adjusted_risk(risk: float, horizon_years: int) -> tuple[float, str]:
    """Shift risk tolerance by investment horizon; returns (effective risk, explanation)."""
    if horizon_years < SHORT_HORIZON_YEARS:
        effective = min(risk, SHORT_HORIZON_RISK_CAP)
        rule = f"under {SHORT_HORIZON_YEARS}-year horizon: risk capped at {SHORT_HORIZON_RISK_CAP:g}"
    else:
        min_years, shift = next((y, s) for y, s in HORIZON_RISK_SHIFTS if horizon_years >= y)
        effective = risk + shift
        upper = min((y for y, _ in HORIZON_RISK_SHIFTS if y > min_years), default=None)
        span = f"{min_years}+ year" if upper is None else f"{min_years}-{upper - 1} year"
        rule = f"{span} horizon: {shift:+g} risk" if shift else f"{span} horizon: no risk adjustment"
    effective = min(max(effective, MIN_RISK), MAX_RISK)
    return effective, f"{rule} ({risk:g} -> {effective:g})"


def risk_band(risk: float) -> str:
    return next(label for upper, label in RISK_BANDS if risk <= upper)
