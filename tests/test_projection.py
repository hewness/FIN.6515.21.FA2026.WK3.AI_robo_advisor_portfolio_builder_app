import pytest

from portfolio_builder.service import ProjectionConfig, project_portfolio_value


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
