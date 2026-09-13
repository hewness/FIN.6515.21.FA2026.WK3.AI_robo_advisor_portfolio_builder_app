"""Locate a portfolio relative to the efficient frontier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import FrontierPosition

ON_FRONTIER_TOLERANCE = 1e-4  # 1 basis point of annual return
_VOL_TOLERANCE = 1e-8


def locate_on_frontier(volatility: float, expected_return: float, frontier_points: pd.DataFrame) -> FrontierPosition:
    """Compare a portfolio with the frontier's return at the same volatility.

    ``return_gap`` is portfolio return minus frontier return (negative = below the frontier).
    ``position`` runs from 0 at the min-volatility end to 1 at the max-return end.
    """
    if frontier_points.empty:
        raise ValueError("Efficient frontier has no points")
    frontier = frontier_points.sort_values("volatility")
    vols = frontier["volatility"].to_numpy(dtype=float)
    rets = frontier["expected_return"].to_numpy(dtype=float)
    vol_min, vol_max = vols[0], vols[-1]

    frontier_return = float(np.interp(volatility, vols, rets))
    gap = expected_return - frontier_return
    span = vol_max - vol_min
    position = 0.0 if span <= _VOL_TOLERANCE else float(np.clip((volatility - vol_min) / span, 0.0, 1.0))
    nearest = int(frontier.index[np.argmin(np.abs(vols - volatility))])

    note = None
    in_range = vol_min - _VOL_TOLERANCE <= volatility <= vol_max + _VOL_TOLERANCE
    if volatility < vol_min - _VOL_TOLERANCE:
        note = (
            f"Volatility {volatility:.2%} is below the frontier's minimum {vol_min:.2%} "
            "(the frontier covers the 11 funds only, not cash); compared with the min-volatility portfolio."
        )
    elif volatility > vol_max + _VOL_TOLERANCE:
        note = (
            f"Volatility {volatility:.2%} is above the frontier's maximum {vol_max:.2%}; "
            "compared with the max-return portfolio."
        )
    elif gap < -ON_FRONTIER_TOLERANCE:
        note = f"{-gap:.2%} per year below the frontier at the same volatility."
    elif gap > ON_FRONTIER_TOLERANCE:
        note = f"{gap:.2%} per year above the funds-only frontier (holds cash at the risk-free rate)."

    return FrontierPosition(
        volatility=volatility,
        expected_return=expected_return,
        frontier_return_at_volatility=frontier_return,
        return_gap=gap,
        position=position,
        nearest_point=nearest,
        on_frontier=in_range and abs(gap) <= ON_FRONTIER_TOLERANCE,
        note=note,
    )
