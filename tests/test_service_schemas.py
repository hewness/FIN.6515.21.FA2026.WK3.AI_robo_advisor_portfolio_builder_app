import pytest
from pydantic import ValidationError

from portfolio_builder.service import FinancialGoal, PortfolioRequest, RiskLevel, get_form_options

BOUNDS = {
    "horizon_years": (1, 30),
    "initial_investment": (1_000, 10_000_000),
    "monthly_contribution": (0, 50_000),
    "age": (18, 80),
    "target_amount": (1_000, 100_000_000),
    "backtest_years": (10, 20),
    "annual_income": (0, 5_000_000),
    "retirement_income": (0, 1_000_000),
    "retirement_age": (50, 75),
}
INT_FIELDS = ("horizon_years", "age", "backtest_years", "retirement_age")


def test_defaults_are_valid():
    req = PortfolioRequest()
    assert req.risk_score == 5.5 and req.goal is FinancialGoal.RETIREMENT
    assert (req.horizon_years, req.initial_investment, req.monthly_contribution, req.age) == (25, 50_000, 1_000, 40)
    assert req.target_amount is None and req.resolved_target == 1_500_000
    assert (req.backtest_years, req.rebalance) == (10, "quarterly")
    assert (req.annual_income, req.retirement_income, req.retirement_age) == (85_000, None, 67)
    assert req.resolved_retirement_income == pytest.approx(34_000) and not req.is_retired


def test_retirement_income_blank_and_override():
    assert PortfolioRequest(annual_income=100_000, retirement_income="").resolved_retirement_income == 40_000
    assert PortfolioRequest(annual_income=100_000, retirement_income=25_000).resolved_retirement_income == 25_000
    assert PortfolioRequest(age=68, retirement_age=67).is_retired


@pytest.mark.parametrize("goal,target", [("retirement", 1_500_000), ("home_purchase", 150_000),
                                         ("education", 200_000), ("general_wealth", 1_000_000)])
def test_goal_target_defaults_and_override(goal, target):
    assert PortfolioRequest(goal=goal).resolved_target == target
    assert PortfolioRequest(goal=goal, target_amount=42_000).resolved_target == 42_000
    assert PortfolioRequest(goal=goal, target_amount="").resolved_target == target


def test_rebalance_values():
    assert PortfolioRequest(rebalance="Monthly").rebalance == "monthly"
    with pytest.raises(ValidationError):
        PortfolioRequest(rebalance="weekly")


@pytest.mark.parametrize("field,bounds", BOUNDS.items())
def test_numeric_bounds(field, bounds):
    low, high = bounds
    assert getattr(PortfolioRequest(**{field: low}), field) == low
    assert getattr(PortfolioRequest(**{field: high}), field) == high
    step = 1 if field in INT_FIELDS else 0.01
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
    assert list(options) == ["risk_tolerance", "horizon_years", "initial_investment", "monthly_contribution", "goal",
                             "age", "target_amount", "backtest_years", "rebalance", "hump_glide_path",
                             "annual_income", "retirement_income", "retirement_age"]
    assert options["annual_income"]["default"] == 85_000 and options["retirement_income"]["default"] is None
    assert options["retirement_income"]["default_rate"] == 0.4
    assert options["hump_glide_path"]["default"] is True and options["hump_glide_path"]["widget"] == "checkbox"
    for field, (low, high) in BOUNDS.items():
        assert (options[field]["minimum"], options[field]["maximum"]) == (low, high)
    assert options["risk_tolerance"]["minimum"] == 1 and options["risk_tolerance"]["maximum"] == 10
    assert [c["score"] for c in options["risk_tolerance"]["choices"]] == [3.0, 5.5, 8.0]
    assert [c["label"] for c in options["goal"]["choices"]] == ["Retirement", "Home purchase", "Education", "General wealth"]
    assert options["goal"]["default"] == "retirement"
    assert [c["default_target"] for c in options["goal"]["choices"]] == [1_500_000, 150_000, 200_000, 1_000_000]
    assert options["target_amount"]["default"] == 1_500_000
    assert [c["value"] for c in options["rebalance"]["choices"]] == ["monthly", "quarterly", "annual"]
    assert options["horizon_years"]["widget"] == "slider"


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("true", True), ("false", False),
                                            ("on", True), (0, False)])
def test_hump_glide_path_coercion(value, expected):
    assert PortfolioRequest(hump_glide_path=value).hump_glide_path is expected
    assert PortfolioRequest().hump_glide_path is True
