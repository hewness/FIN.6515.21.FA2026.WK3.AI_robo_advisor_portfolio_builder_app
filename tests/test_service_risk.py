import pytest

from portfolio_builder.service import horizon_adjusted_risk, resolve_risk_tolerance, risk_band


@pytest.mark.parametrize(
    "years,risk,expected",
    [
        (1, 8.0, 3.0), (2, 8.0, 3.0), (2, 2.0, 2.0),   # under 3 years: capped at 3
        (3, 5.0, 3.0), (4, 5.0, 3.0),                  # 3-4 years: -2
        (5, 5.0, 4.0), (9, 5.0, 4.0),                  # 5-9 years: -1
        (10, 5.0, 5.0), (19, 5.0, 5.0),                # 10-19 years: 0
        (20, 5.0, 6.0), (30, 5.0, 6.0),                # 20+ years: +1
    ],
)
def test_horizon_adjustment_table(years, risk, expected):
    effective, note = horizon_adjusted_risk(risk, years)
    assert effective == expected
    assert f"-> {expected:g}" in note


def test_adjusted_risk_stays_in_range():
    assert horizon_adjusted_risk(10.0, 25)[0] == 10.0
    assert horizon_adjusted_risk(1.0, 4)[0] == 1.0


def test_notes_describe_rule():
    assert "capped at 3" in horizon_adjusted_risk(8, 2)[1]
    assert "20+ year horizon: +1" in horizon_adjusted_risk(5, 25)[1]
    assert "3-4 year horizon: -2" in horizon_adjusted_risk(5, 3)[1]
    assert "5-9 year horizon: -1" in horizon_adjusted_risk(5, 7)[1]
    assert "10-19 year horizon: no risk adjustment" in horizon_adjusted_risk(5, 12)[1]


def test_resolve_and_bands():
    assert resolve_risk_tolerance("Aggressive") == 8.0
    with pytest.raises(ValueError):
        resolve_risk_tolerance(11)
    assert [risk_band(r) for r in (1, 3.5, 3.6, 7, 7.5, 10)] == [
        "Conservative", "Conservative", "Moderate", "Moderate", "Aggressive", "Aggressive"
    ]
