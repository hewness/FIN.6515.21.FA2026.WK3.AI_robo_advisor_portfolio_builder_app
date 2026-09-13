import itertools

import numpy as np
import pandas as pd
import pytest

from portfolio_builder.optimization import InvestorProfile, MarketInputs, MeanVarianceStrategy, OptimizationError
from portfolio_builder.optimization import mean_variance as mv

TICKERS = ["SPY", "EFA", "AGG", "TIP", "VNQ"]


def synthetic_inputs(rf: float = 0.03) -> MarketInputs:
    mu = np.array([0.09, 0.07, 0.035, 0.03, 0.08])
    vol = np.array([0.16, 0.18, 0.05, 0.06, 0.20])
    corr = np.array(
        [
            [1.0, 0.8, 0.1, 0.0, 0.6],
            [0.8, 1.0, 0.1, 0.0, 0.5],
            [0.1, 0.1, 1.0, 0.7, 0.2],
            [0.0, 0.0, 0.7, 1.0, 0.1],
            [0.6, 0.5, 0.2, 0.1, 1.0],
        ]
    )
    cov = corr * np.outer(vol, vol)
    return MarketInputs(pd.Series(mu, index=TICKERS), pd.DataFrame(cov, index=TICKERS, columns=TICKERS), rf)


def assert_valid(w, max_weight=1.0):
    assert np.all(w >= 0)
    assert np.all(w <= max_weight + 1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def vol(w, cov):
    return float(np.sqrt(w @ cov @ w))


def test_min_volatility_two_asset_closed_form():
    s1, s2, rho = 0.2, 0.1, 0.3
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w = mv.min_volatility(np.array([0.1, 0.05]), cov)
    expected_w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    assert w[0] == pytest.approx(expected_w1, abs=1e-6)
    assert_valid(w)


def test_max_sharpe_matches_grid_search():
    inputs = synthetic_inputs()
    idx = [0, 2, 4]  # 3-asset subproblem for a tractable grid
    mu = inputs.expected_returns.to_numpy()[idx]
    cov = inputs.covariance.to_numpy()[np.ix_(idx, idx)]
    w = mv.max_sharpe(mu, cov, inputs.risk_free_rate)
    assert_valid(w)

    best = -np.inf
    steps = np.linspace(0, 1, 201)
    for a, b in itertools.product(steps, steps):
        if a + b <= 1:
            g = np.array([a, b, 1 - a - b])
            best = max(best, (g @ mu - inputs.risk_free_rate) / vol(g, cov))
    assert (w @ mu - inputs.risk_free_rate) / vol(w, cov) >= best - 1e-4


def test_max_return_respects_max_weight():
    mu = np.array([0.05, 0.10, 0.08])
    assert list(mv.max_return(mu)) == [0, 1, 0]
    np.testing.assert_allclose(mv.max_return(mu, max_weight=0.4), [0.2, 0.4, 0.4])
    with pytest.raises(OptimizationError):
        mv.max_return(mu, max_weight=0.3)


def test_efficient_frontier_is_monotone_and_valid():
    inputs = synthetic_inputs()
    frontier = mv.efficient_frontier(inputs, n_points=25)
    assert len(frontier) == 25
    pts = frontier.points
    assert pts["expected_return"].is_monotonic_increasing
    assert (pts["volatility"].diff().dropna() >= -1e-8).all()
    assert pts["volatility"].min() == pytest.approx(pts["volatility"].iloc[0])
    for _, row in frontier.weights.iterrows():
        assert_valid(row.to_numpy())
    assert list(frontier.to_frame().columns[:3]) == ["expected_return", "volatility", "sharpe_ratio"]


def test_efficient_risk_hits_target_and_handles_out_of_range():
    inputs = synthetic_inputs()
    mu, cov = inputs.expected_returns.to_numpy(), inputs.covariance.to_numpy()
    w, note = mv.efficient_risk(mu, cov, 0.10)
    assert_valid(w)
    assert note is None
    assert vol(w, cov) == pytest.approx(0.10, abs=1e-5)  # constraint binds on the efficient frontier

    w_low, note_low = mv.efficient_risk(mu, cov, 0.001)
    np.testing.assert_allclose(w_low, mv.min_volatility(mu, cov), atol=1e-8)
    assert "min-volatility" in note_low

    w_high, note_high = mv.efficient_risk(mu, cov, 5.0)
    np.testing.assert_allclose(w_high, mv.max_return(mu))
    assert "max-return" in note_high


def test_efficient_return_hits_target_and_rejects_infeasible():
    inputs = synthetic_inputs()
    mu, cov = inputs.expected_returns.to_numpy(), inputs.covariance.to_numpy()
    w = mv.efficient_return(mu, cov, 0.06)
    assert_valid(w)
    assert w @ mu == pytest.approx(0.06, abs=1e-6)
    with pytest.raises(OptimizationError):
        mv.efficient_return(mu, cov, 0.20)


@pytest.mark.parametrize("max_weight", [1.0, 0.4])
def test_all_solvers_long_only_fully_invested(max_weight):
    inputs = synthetic_inputs()
    mu, cov = inputs.expected_returns.to_numpy(), inputs.covariance.to_numpy()
    assert_valid(mv.min_volatility(mu, cov, max_weight), max_weight)
    assert_valid(mv.max_sharpe(mu, cov, inputs.risk_free_rate, max_weight), max_weight)
    assert_valid(mv.efficient_risk(mu, cov, 0.09, max_weight)[0], max_weight)
    frontier = mv.efficient_frontier(inputs, 10, max_weight)
    for _, row in frontier.weights.iterrows():
        assert_valid(row.to_numpy(), max_weight)


def test_strategy_volatility_increases_with_risk_tolerance():
    inputs = synthetic_inputs()
    strategy = MeanVarianceStrategy(n_points=20)
    vols = [strategy.allocate(InvestorProfile(40, r), inputs).metrics.volatility for r in range(1, 11)]
    assert all(b >= a - 1e-6 for a, b in zip(vols, vols[1:]))

    low = strategy.allocate(InvestorProfile(40, 1), inputs)
    high = strategy.allocate(InvestorProfile(40, 10), inputs)
    assert low.metrics.volatility == pytest.approx(low.reference_portfolios["min_volatility"].metrics.volatility, abs=1e-6)
    assert high.metrics.volatility == pytest.approx(high.reference_portfolios["max_return"].metrics.volatility, abs=1e-6)


def test_strategy_result_contents():
    inputs = synthetic_inputs()
    result = MeanVarianceStrategy(n_points=15).allocate(InvestorProfile(35, 6), inputs)
    assert result.method == "mean_variance"
    assert len(result.frontier) == 15
    assert set(result.reference_portfolios) == {"min_volatility", "max_sharpe", "max_return"}
    assert result.details["target_volatility"] == pytest.approx(result.metrics.volatility, abs=1e-5)
    frontier_vols = result.frontier.points["volatility"]
    assert frontier_vols.min() - 1e-9 <= result.metrics.volatility <= frontier_vols.max() + 1e-9
    max_sharpe = result.reference_portfolios["max_sharpe"].metrics.sharpe_ratio
    assert max_sharpe >= result.frontier.points["sharpe_ratio"].max() - 1e-6


@pytest.mark.parametrize(
    "objective,target,check",
    [
        ("max_sharpe", None, lambda r: r.metrics.sharpe_ratio == pytest.approx(r.reference_portfolios["max_sharpe"].metrics.sharpe_ratio)),
        ("min_volatility", None, lambda r: r.metrics.volatility == pytest.approx(r.reference_portfolios["min_volatility"].metrics.volatility)),
        ("target_volatility", 0.08, lambda r: r.metrics.volatility == pytest.approx(0.08, abs=1e-5)),
        ("target_return", 0.05, lambda r: r.metrics.expected_return == pytest.approx(0.05, abs=1e-6)),
    ],
)
def test_strategy_objectives(objective, target, check):
    result = MeanVarianceStrategy(objective=objective, target=target, n_points=10).allocate(
        InvestorProfile(50, 5), synthetic_inputs()
    )
    assert check(result)


def test_strategy_argument_validation():
    with pytest.raises(ValueError):
        MeanVarianceStrategy(objective="target_volatility")
    with pytest.raises(ValueError):
        MeanVarianceStrategy(objective="bogus")
    with pytest.raises(ValueError):
        MeanVarianceStrategy(vol_range=(0.8, 0.2))
