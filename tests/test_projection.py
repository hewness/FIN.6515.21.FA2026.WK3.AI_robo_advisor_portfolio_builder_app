import math

import pytest

from portfolio_builder.service import ProjectionConfig, project_portfolio_value
from portfolio_builder.service.projection import _future_value

SIMPLE = "simple_percentiles"


def test_zero_volatility_matches_closed_form():
    initial, monthly, years, mu = 10_000.0, 250.0, 10, 0.06
    projection = project_portfolio_value(initial, monthly, years, mu, 0.0, ProjectionConfig(simulations=50))

    g = (1 + mu) ** (1 / 12) - 1
    n = years * 12
    closed_form = initial * (1 + g) ** n + monthly * ((1 + g) ** n - 1) / g
    assert projection.final_expected == pytest.approx(closed_form, abs=0.01)
    for point in projection.points:
        assert point.p10 == point.p50 == point.p90 == pytest.approx(point.expected, abs=0.01)


def test_structure_and_contributions():
    projection = project_portfolio_value(5_000, 100, 7, 0.07, 0.15, ProjectionConfig(simulations=500))
    assert [p.year for p in projection.points] == list(range(8))
    assert projection.points[0].expected == 5_000
    assert projection.points[3].total_contributed == 5_000 + 100 * 36
    assert projection.total_contributed == 5_000 + 100 * 84
    assert projection.simulations == 500


def test_percentiles_ordered_and_spread_grows():
    projection = project_portfolio_value(100_000, 1_000, 20, 0.08, 0.16, ProjectionConfig(simulations=2000))
    for p in projection.points[1:]:
        assert p.p10 < p.p50 < p.p90
    spreads = [p.p90 - p.p10 for p in projection.points]
    assert spreads == sorted(spreads)
    # Lognormal simulation is centered on the expected path (mean), median below it.
    assert projection.final_p50 < projection.final_expected < projection.final_p90


def test_seed_reproducible():
    cfg = ProjectionConfig(simulations=300, seed=7)
    a = project_portfolio_value(1_000, 50, 5, 0.05, 0.1, cfg)
    b = project_portfolio_value(1_000, 50, 5, 0.05, 0.1, cfg)
    assert a == b


@pytest.mark.parametrize("kwargs", [{"years": 0}, {"expected_return": -1.0}, {"volatility": -0.1}])
def test_rejects_invalid(kwargs):
    args = {"initial": 1_000, "monthly_contribution": 0, "years": 5, "expected_return": 0.05, "volatility": 0.1}
    with pytest.raises(ValueError):
        project_portfolio_value(**{**args, **kwargs})


def test_all_percentiles_ordered():
    projection = project_portfolio_value(20_000, 500, 15, 0.07, 0.14, ProjectionConfig(simulations=1500))
    for p in projection.points[1:]:
        assert p.p10 < p.p25 < p.p50 < p.p75 < p.p90
    assert projection.final_p25 == projection.points[-1].p25 and projection.final_p75 == projection.points[-1].p75


def test_probability_of_meeting_target():
    cfg = ProjectionConfig(simulations=2000)
    base = dict(initial=100_000, monthly_contribution=1_000, years=20, expected_return=0.07, volatility=0.15, config=cfg)
    assert project_portfolio_value(**base).probability_of_meeting_target is None
    assert project_portfolio_value(**base, target_amount=1_000).probability_of_meeting_target == 1.0
    assert project_portfolio_value(**base, target_amount=1e9).probability_of_meeting_target == 0.0
    median = project_portfolio_value(**base, target_amount=None).final_p50
    mid = project_portfolio_value(**base, target_amount=median).probability_of_meeting_target
    assert mid == pytest.approx(0.5, abs=0.01)
    low = project_portfolio_value(**base, target_amount=median * 0.8).probability_of_meeting_target
    assert low > mid


def test_probability_zero_volatility_is_binary():
    proj = project_portfolio_value(10_000, 0, 10, 0.05, 0.0, ProjectionConfig(simulations=10), target_amount=16_000)
    assert proj.probability_of_meeting_target == 1.0  # 10,000 * 1.05^10 = 16,289
    proj = project_portfolio_value(10_000, 0, 10, 0.05, 0.0, ProjectionConfig(simulations=10), target_amount=16_500)
    assert proj.probability_of_meeting_target == 0.0


def test_monte_carlo_keeps_rank_spaced_sample_paths():
    projection = project_portfolio_value(50_000, 500, 20, 0.08, 0.16, ProjectionConfig(simulations=2000))
    assert projection.method == "monte_carlo" and len(projection.sample_paths) == 30
    assert all(len(path) == 21 and path[0] == 50_000 for path in projection.sample_paths)
    finals = [path[-1] for path in projection.sample_paths]
    assert finals == sorted(finals)  # ordered worst to best
    assert finals[0] <= projection.final_p10 and finals[-1] >= projection.final_p90
    assert len(project_portfolio_value(1_000, 0, 3, 0.05, 0.1, ProjectionConfig(simulations=10)).sample_paths) == 10


def test_simple_percentiles_zero_volatility_matches_closed_form():
    initial, monthly, years, mu = 10_000.0, 250.0, 10, 0.06
    projection = project_portfolio_value(initial, monthly, years, mu, 0.0, method=SIMPLE)
    g = (1 + mu) ** (1 / 12) - 1
    n = years * 12
    closed_form = initial * (1 + g) ** n + monthly * ((1 + g) ** n - 1) / g
    assert projection.final_p10 == projection.final_p90 == pytest.approx(closed_form, abs=0.01)


def test_simple_percentiles_formula_and_structure():
    mu, vol, years = 0.08, 0.16, 20
    projection = project_portfolio_value(100_000, 1_000, years, mu, vol, method=SIMPLE)
    assert projection.method == SIMPLE and projection.simulations == 0 and projection.seed is None
    assert projection.sample_paths == []
    for p in projection.points[1:]:
        assert p.p10 < p.p25 < p.p50 < p.p75 < p.p90
    assert projection.final_p50 < projection.final_expected
    sigma_log = math.sqrt(math.log1p(vol**2 / (1 + mu) ** 2))
    mu_log = math.log1p(mu) - sigma_log**2 / 2
    assert projection.final_p50 == pytest.approx(_future_value(100_000, 1_000, math.exp(mu_log) - 1, years * 12), abs=0.01)
    # deterministic: no simulation, so the seed doesn't matter
    other = project_portfolio_value(100_000, 1_000, years, mu, vol, ProjectionConfig(seed=1, simulations=10), method=SIMPLE)
    assert other.points == projection.points


def test_simple_percentiles_goal_probability():
    base = dict(initial=100_000, monthly_contribution=1_000, years=20, expected_return=0.07, volatility=0.15, method=SIMPLE)
    median = project_portfolio_value(**base).final_p50
    assert project_portfolio_value(**base, target_amount=median).probability_of_meeting_target == pytest.approx(0.5, abs=1e-4)
    assert project_portfolio_value(**base, target_amount=1_000).probability_of_meeting_target == 1.0
    assert project_portfolio_value(**base, target_amount=1e12).probability_of_meeting_target == pytest.approx(0.0, abs=1e-9)
    odds = [project_portfolio_value(**base, target_amount=t).probability_of_meeting_target
            for t in (median * 0.6, median * 0.9, median * 1.2, median * 2)]
    assert odds == sorted(odds, reverse=True)
    zero_vol = {**base, "volatility": 0.0, "monthly_contribution": 0, "initial": 10_000, "years": 10, "expected_return": 0.05}
    assert project_portfolio_value(**zero_vol, target_amount=16_000).probability_of_meeting_target == 1.0
    assert project_portfolio_value(**zero_vol, target_amount=16_500).probability_of_meeting_target == 0.0


def test_simple_percentiles_close_to_monte_carlo():
    args = dict(initial=100_000, monthly_contribution=500, years=15, expected_return=0.07, volatility=0.12,
                target_amount=400_000)
    mc = project_portfolio_value(**args, config=ProjectionConfig(simulations=5000))
    simple = project_portfolio_value(**args, method=SIMPLE)
    assert simple.final_p50 == pytest.approx(mc.final_p50, rel=0.03)
    assert simple.probability_of_meeting_target == pytest.approx(mc.probability_of_meeting_target, abs=0.05)


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        project_portfolio_value(1_000, 0, 5, 0.05, 0.1, method="bootstrap")
