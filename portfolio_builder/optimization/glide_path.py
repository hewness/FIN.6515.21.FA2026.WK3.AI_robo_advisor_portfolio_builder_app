"""Equity glide paths: how the equity share of a portfolio changes with the investor's age."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import InvestorProfile

DEFAULT_MAX_RISK_SHIFT = 20.0  # percentage points at risk tolerance 1 (-) and 10 (+)
DEFAULT_MIN_EQUITY = 0.10
DEFAULT_MAX_EQUITY = 1.00


class GlidePath(ABC):
    """Base equity share (0-1) by age, before any risk-tolerance adjustment."""

    name: str
    label: str

    @abstractmethod
    def base_equity(self, age: float) -> float:
        ...


@dataclass(frozen=True)
class LinearGlidePath(GlidePath):
    """The classic rule: equity % = base - age (monotonically declining)."""

    base: float = 110.0
    name: str = "linear"

    @property
    def label(self) -> str:
        return f"Linear ({self.base:g} − age)"

    def base_equity(self, age: float) -> float:
        return (self.base - age) / 100.0


@dataclass(frozen=True)
class HumpGlidePath(GlidePath):
    """Hump-shaped path: piecewise-linear through (age, equity) points, flat before the first and after the last."""

    points: tuple[tuple[float, float], ...] = ((25.0, 0.60), (45.0, 0.80), (65.0, 0.60))
    name: str = "hump"

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("A hump glide path needs at least two points")
        ages = [a for a, _ in self.points]
        if any(b <= a for a, b in zip(ages, ages[1:])):
            raise ValueError("Glide path ages must be strictly increasing")
        if any(not 0.0 <= e <= 1.0 for _, e in self.points):
            raise ValueError("Glide path equity shares must be between 0 and 1")

    @property
    def label(self) -> str:
        return "Hump-shaped"

    @property
    def peak(self) -> tuple[float, float]:
        return max(self.points, key=lambda p: p[1])

    def base_equity(self, age: float) -> float:
        ages, equities = zip(*self.points)
        return float(np.interp(age, ages, equities))

    def describe(self) -> str:
        (first_age, first_eq), (last_age, last_eq) = self.points[0], self.points[-1]
        peak_age, peak_eq = self.peak
        return (f"{first_eq:.0%} until age {first_age:g}, rising to {peak_eq:.0%} at {peak_age:g}, "
                f"then {last_eq:.0%} from {last_age:g} on")


GLIDE_PATHS: dict[str, GlidePath] = {"linear": LinearGlidePath(), "hump": HumpGlidePath()}


def get_glide_path(name: str) -> GlidePath:
    try:
        return GLIDE_PATHS[name]
    except KeyError:
        raise KeyError(f"Unknown glide path {name!r}; available: {sorted(GLIDE_PATHS)}") from None


def equity_target(
    glide_path: GlidePath,
    profile: InvestorProfile,
    max_risk_shift: float = DEFAULT_MAX_RISK_SHIFT,
    min_equity: float = DEFAULT_MIN_EQUITY,
    max_equity: float = DEFAULT_MAX_EQUITY,
) -> tuple[float, float]:
    """Equity share for a profile: glide-path base, shifted by risk tolerance, clamped.

    Risk tolerance 1 shifts the base by ``-max_risk_shift`` percentage points and 10 by ``+max_risk_shift``,
    linearly in between. Returns ``(equity fraction, shift in percentage points)``.
    """
    shift = -max_risk_shift + 2 * max_risk_shift * profile.risk_fraction
    raw = glide_path.base_equity(profile.age) + shift / 100.0
    return min(max(raw, min_equity), max_equity), shift


def hump_points_summary(points: Sequence[tuple[float, float]]) -> str:
    return " → ".join(f"{e:.0%} @ {a:g}" for a, e in points)
