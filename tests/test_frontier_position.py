import pandas as pd
import pytest

from portfolio_builder.service import locate_on_frontier

FRONTIER = pd.DataFrame(
    {"expected_return": [0.03, 0.05, 0.07, 0.09], "volatility": [0.04, 0.06, 0.10, 0.16]},
    index=pd.RangeIndex(4, name="point"),
)


def test_point_on_frontier():
    pos = locate_on_frontier(0.08, 0.06, FRONTIER)  # halfway between points 1 and 2
    assert pos.frontier_return_at_volatility == pytest.approx(0.06)
    assert pos.on_frontier and pos.return_gap == pytest.approx(0.0)
    assert pos.position == pytest.approx((0.08 - 0.04) / 0.12)
    assert pos.note is None


def test_point_below_frontier():
    pos = locate_on_frontier(0.10, 0.05, FRONTIER)
    assert pos.return_gap == pytest.approx(-0.02)
    assert not pos.on_frontier
    assert pos.nearest_point == 2
    assert "below the frontier" in pos.note


def test_ends_and_out_of_range():
    assert locate_on_frontier(0.04, 0.03, FRONTIER).position == 0.0
    assert locate_on_frontier(0.16, 0.09, FRONTIER).position == 1.0

    low = locate_on_frontier(0.02, 0.035, FRONTIER)  # e.g. a portfolio with cash
    assert low.position == 0.0 and not low.on_frontier
    assert low.frontier_return_at_volatility == pytest.approx(0.03)
    assert "below the frontier's minimum" in low.note

    high = locate_on_frontier(0.20, 0.08, FRONTIER)
    assert high.position == 1.0 and "above the frontier's maximum" in high.note


def test_unsorted_frontier_and_empty():
    shuffled = FRONTIER.sample(frac=1, random_state=3)
    assert locate_on_frontier(0.08, 0.06, shuffled).on_frontier
    with pytest.raises(ValueError):
        locate_on_frontier(0.1, 0.1, FRONTIER.iloc[0:0])
