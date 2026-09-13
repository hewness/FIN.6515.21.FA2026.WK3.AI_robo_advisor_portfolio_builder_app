import pytest
from pydantic import ValidationError

from portfolio_builder.service import FinancialGoal, PortfolioRequest, RiskLevel, get_form_options

BOUNDS = {
    "horizon_years": (1, 30),
    "initial_investment": (1_000, 10_000_000),
    "monthly_contribution": (0, 50_000),
    "age": (18, 80),
}


def test_defaults_are_valid():
    req = PortfolioRequest()
    assert req.risk_score == 5.5 and req.goal is FinancialGoal.GENERAL_WEALTH


@pytest.mark.parametrize("field,bounds", BOUNDS.items())
def test_numeric_bounds(field, bounds):
    low, high = bounds
    assert getattr(PortfolioRequest(**{field: low}), field) == low
    assert getattr(PortfolioRequest(**{field: high}), field) == high
    step = 1 if field in ("horizon_years", "age") else 0.01
    for bad in (low - step, high + step):
        with pytest.raises(ValidationError):
            PortfolioRequest(**{field: bad})


@pytest.mark.parametrize("value,score", [("Conservative", 3.0), ("MODERATE", 5.5), (" aggressive ", 8.0),
                                         (RiskLevel.MODERATE, 5.5), (1, 1.0), (10, 10.0), ("7.5", 7.5)])
def test_risk_tolerance_accepts_labels_and_numbers(value, score):
    assert PortfolioRequest(risk_tolerance=value).risk_score == score


@pytest.mark.parametrize("value", ["yolo", 0.5, 10.5, "", True, None])
def test_risk_tolerance_rejects_invalid(value):
    with pytest.raises(ValidationError, match="Risk tolerance"):
        PortfolioRequest(risk_tolerance=value)


@pytest.mark.parametrize("value", ["home_purchase", "Home purchase", "HOME-PURCHASE", FinancialGoal.HOME_PURCHASE])
def test_goal_accepts_values_and_labels(value):
    assert PortfolioRequest(goal=value).goal is FinancialGoal.HOME_PURCHASE


def test_coerces_widget_values_and_forbids_unknown_fields():
    req = PortfolioRequest(age="45", horizon_years=20.0, initial_investment="250000")
    assert req.age == 45 and req.horizon_years == 20 and req.initial_investment == 250_000
    with pytest.raises(ValidationError):
        PortfolioRequest(nickname="x")


def test_form_options_match_model():
    options = get_form_options()
    assert list(options) == ["risk_tolerance", "horizon_years", "initial_investment", "monthly_contribution", "goal", "age"]
    for field, (low, high) in BOUNDS.items():
        assert (options[field]["minimum"], options[field]["maximum"]) == (low, high)
    assert options["risk_tolerance"]["minimum"] == 1 and options["risk_tolerance"]["maximum"] == 10
    assert [c["score"] for c in options["risk_tolerance"]["choices"]] == [3.0, 5.5, 8.0]
    assert [c["label"] for c in options["goal"]["choices"]] == ["Retirement", "Home purchase", "Education", "General wealth"]
    assert options["goal"]["default"] == "general_wealth"
    assert options["horizon_years"]["widget"] == "slider"
