import pytest

from portfolio_builder.optimization import InvestorProfile
from portfolio_builder.optimization.glide_path import (
    HumpGlidePath,
    LinearGlidePath,
    equity_target,
    get_glide_path,
)


@pytest.mark.parametrize("age,expected", [(18, .60), (25, .60), (35, .70), (45, .80), (55, .70), (65, .60), (80, .60),
                                          (100, .60)])
def test_hump_shape(age, expected):
    assert HumpGlidePath().base_equity(age) == pytest.approx(expected)


def test_hump_rises_then_falls_then_flat():
    path = HumpGlidePath()
    values = [path.base_equity(a) for a in range(18, 81)]
    peak = values.index(max(values))
    assert 18 + peak == 45
    assert all(b >= a for a, b in zip(values[:peak], values[1:peak + 1]))      # non-decreasing to the peak
    assert all(b <= a for a, b in zip(values[peak:], values[peak + 1:]))       # non-increasing after
    assert len({round(v, 12) for v in values[65 - 18:]}) == 1                   # flat from 65


@pytest.mark.parametrize("age", [18, 30, 45, 60, 80])
def test_linear_is_classic_rule(age):
    assert LinearGlidePath().base_equity(age) == pytest.approx((110 - age) / 100)
    assert LinearGlidePath(base=120).base_equity(age) == pytest.approx((120 - age) / 100)


@pytest.mark.parametrize("risk,shift", [(1, -20.0), (5.5, 0.0), (10, 20.0)])
def test_equity_target_applies_risk_shift(risk, shift):
    equity, points = equity_target(HumpGlidePath(), InvestorProfile(45, risk))
    assert points == pytest.approx(shift)
    assert equity == pytest.approx(0.80 + shift / 100)


def test_equity_target_clamps():
    assert equity_target(LinearGlidePath(), InvestorProfile(18, 10))[0] == 1.0      # 92% + 20 -> capped
    assert equity_target(LinearGlidePath(), InvestorProfile(100, 1))[0] == 0.10     # -10% -> floor
    assert equity_target(HumpGlidePath(), InvestorProfile(45, 6.5))[0] == pytest.approx(0.8444, abs=1e-4)


@pytest.mark.parametrize("points", [((45, .8),), ((45, .8), (25, .6)), ((25, .6), (45, 1.2))])
def test_invalid_hump_points(points):
    with pytest.raises(ValueError):
        HumpGlidePath(points=points)


def test_registry_and_labels():
    assert get_glide_path("hump").label == "Hump-shaped"
    assert get_glide_path("linear").label.startswith("Linear")
    assert "80% at 45" in HumpGlidePath().describe()
    with pytest.raises(KeyError):
        get_glide_path("s-curve")
