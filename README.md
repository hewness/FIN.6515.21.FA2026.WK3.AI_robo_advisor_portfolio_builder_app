---
title: AI Robo-Advisor Portfolio Builder
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# AI Robo-Advisor Portfolio Builder

FIN.6515.21 FA2026 Week 3: a Gradio app for building AI robo-advisor portfolios.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860.

## Web app

`app.py` launches a Gradio dashboard (`portfolio_builder/ui`) that compares the rule-based and mean-variance portfolios side by side.

**Sidebar inputs** (defaults in brackets). The dashboard updates when you release a slider, change a choice, or leave a text field. Money fields format as you type (`$#,000`) and accept shorthand such as `50k` or `1.2m`. Hover over a label's ⓘ to see what the input does and its allowed range.

| Group | Input | Control | Range / options | Default |
|---|---|---|---|---|
| About you | Age | Number | 18–80 | 40 |
| | Financial goal | Dropdown | Retirement / Home purchase / Education / General wealth | Retirement |
| | Goal target | Money text | $1,000–$100M; resets to the goal's default when the goal changes | $1.5M (home $150k, education $200k, general $1M) |
| Risk profile | Risk tolerance | Preset (Conservative / Moderate / Aggressive / Custom) + 1–10 slider | presets = 3 / 5.5 / 8 | Moderate (5.5) |
| Investment plan | Initial investment | Money text | $1,000–$10,000,000 | $50,000 |
| | Monthly contribution | Money text | $0–$50,000 | $1,000 |
| | Investment horizon | Slider | 1–30 years | 25 |
| Backtest settings | Lookback | Slider | 10–20 years | 15 |
| | Rebalancing | Radio | Monthly / Quarterly / Annual | Quarterly |

Invalid inputs are listed in a red status box in the sidebar, and the charts keep the last valid results. Input combinations worth a second look (an unreachable target, a horizon beyond age 75, a very long home or education horizon, a capped short horizon) show an amber warning.

**Main panel**
- **A card for each portfolio** (Rule-based Lifecycle Portfolio in indigo, Mean-variance Optimized Portfolio in teal), side by side:
  - Tiles for expected annual return, volatility, Sharpe ratio, maximum historical drawdown (from the backtest), and probability of reaching the goal (share of 5,000 simulated outcomes at or above the target).
  - An allocation-by-asset-class donut next to the holdings table. The table is tall enough for every holding (up to 11 funds plus cash), so it never scrolls, and each section lines up across the two cards. Each holding row starts with a color swatch that matches its asset class's slice. Hovering over (or tapping) a slice highlights that asset class's holdings and dims the rest.
  - Projected wealth: expected (mean), optimistic (75th percentile) and pessimistic (25th percentile) paths, including contributions, with the goal target line. Both cards use the same scale.
- **Rule-based Lifecycle Portfolio vs Mean-variance Optimized Portfolio:** a comparison container with two charts side by side that each plot both portfolios.
  - **Risk vs. Return:** asset classes (each an equal-weight blend of its funds), cash, the efficient frontier, the max-Sharpe point and both portfolios.
  - **Historical Backtest vs. S&P 500 (SPY):** growth of the initial investment and drawdowns over 10–20 years.
- **Notes & assumptions**, including the full name of each fund.

**Colors:**
- Indigo, teal and slate always mean the rule-based portfolio, the mean-variance portfolio and the S&P 500 benchmark, in every widget.
- Asset classes use their own palette so they never look like a portfolio: warm tones for stocks, greens for bonds, warm gray for cash.

**Backtest method** (`portfolio_builder/backtest`):
- Daily total returns with fixed target weights. Holdings drift and reset at each rebalance date. Cash earns the risk-free rate.
- Before a fund existed, its sibling in the same asset class stands in (VTI→SPY, VXUS→EFA, VWO→EEM, BND→AGG, VTIP→TIP), and the chart caption says so.
- It models the initial investment only, with no contributions, fees or taxes.
- Mean-variance weights are estimated on overlapping history, so that backtest is in-sample.

## Investment universe

| Asset class | Role in portfolio | Tickers |
|---|---|---|
| US large-cap stocks | Growth, domestic equity exposure | SPY, VTI |
| International developed stocks | Diversification, international exposure | EFA, VXUS |
| Emerging market stocks | Higher growth potential, higher risk | EEM, VWO |
| US Aggregate bonds | Stability, income | AGG, BND |
| Treasury inflation-protected securities | Inflation Hedge | TIP, VTIP |
| Real estate (REITs) | Real asset diversification | VNQ |
| Cash and Money Markets | Liquidity, capital preservation | *(no ticker)* |

Defined in `portfolio_builder/universe.py`.

## Market data

Daily history comes from Yahoo Finance (`yfinance`) and is cached on disk under `data/market_data/`:

- `prices/<TICKER>.parquet`: full history with `open, high, low, close, adj_close, volume, dividends, stock_splits, capital_gains`
- `info/<TICKER>.json`: fund name, category, expense ratio (in percent) and similar fields
- `manifest.json`: when each ticker was downloaded and the date range it covers

A snapshot of the cache is committed to the repo (the Parquet files are stored with Git LFS). Data is read from disk when it's there and downloaded when it isn't. Set `MARKET_DATA_CACHE_DIR` to use a different folder.

### Command line

```bash
python -m portfolio_builder.market_data status                  # what is cached
python -m portfolio_builder.market_data download                # download missing tickers only
python -m portfolio_builder.market_data refresh                 # force re-download of everything
python -m portfolio_builder.market_data refresh --tickers SPY   # force re-download of one ticker
```

### Python

```python
from portfolio_builder.market_data import get_market_data_service

svc = get_market_data_service()
prices  = svc.get_prices()                          # adjusted close, dates x tickers, common dates only
returns = svc.get_returns(frequency="monthly")      # total returns; also "daily"/"weekly", method="log"
cov     = returns.cov() * 12                        # annualized covariance for optimization
divs    = svc.get_dividends(["AGG", "BND"])         # cash dividends per share
meta    = svc.get_universe_metadata()               # asset class, role, name, expense ratio
svc.refresh(["SPY"])                                # re-download; keeps old data if the download fails
```

Every method defaults to the full universe. Use `align="outer"` to keep each fund's full history (older funds start before VTIP's 2012 inception). A data source other than yfinance can be plugged in by implementing `MarketDataConnector` (`portfolio_builder/market_data/connectors/base.py`).

## Portfolio optimization

`portfolio_builder/optimization` turns a client profile into a portfolio. A profile is an age plus a **risk tolerance from 1 (most conservative) to 10 (most aggressive)**. Expected returns and covariance are estimated from the cached monthly total returns over the window where all funds have data, then annualized. Every portfolio is long-only (no shorting) and its weights sum to 1; results that break either rule are rejected.

| Approach | How it allocates |
|---|---|
| `rule_based` | Equity % = 110 − age, shifted from −20 points (risk 1) to +20 points (risk 10) and kept between 10% and 100%. Equity is split 55% US large-cap / 25% international developed / 10% emerging / 10% REITs. The rest is split 70% aggregate bonds / 20% TIPS / 10% cash. Within each asset class, the tickers are ranked by historical volatility: conservative clients hold the calmer fund, aggressive clients the more volatile one, and clients in between get a blend. |
| `mean_variance` | Builds the long-only efficient frontier with `scipy.optimize.minimize` (SLSQP). The client portfolio has the highest return for a target volatility, placed by risk tolerance between the minimum-volatility portfolio (risk 1) and the maximum-return portfolio (risk 10). Other objectives: `max_sharpe`, `min_volatility`, `target_volatility`, `target_return`. The result also includes the frontier and the min-volatility, max-Sharpe and max-return reference portfolios. |

Cash (shown as `CASH`) earns the risk-free rate, default 4%, at zero volatility. Only the rule-based approach holds it.

```python
from portfolio_builder.optimization import InvestorProfile, get_optimization_engine

engine = get_optimization_engine(risk_free_rate=0.04)
profile = InvestorProfile(age=40, risk_tolerance=6)

rule = engine.optimize("rule_based", profile)
mvo = engine.optimize("mean_variance", profile)
mvo.weights, mvo.metrics, mvo.asset_class_weights
mvo.frontier.points                                   # expected_return, volatility, sharpe_ratio
mvo.reference_portfolios["max_sharpe"].weights
engine.optimize("mean_variance", profile, objective="target_volatility", target=0.10, max_weight=0.4)
engine.compare(profile)                               # all registered approaches
```

To add a new approach (e.g. risk parity), subclass `AllocationStrategy`, implement `allocate(profile, inputs)`, and call `engine.register(...)`.

```bash
python -m portfolio_builder.optimization --age 40 --risk 6                       # both approaches
python -m portfolio_builder.optimization --age 30 --risk 8 --method mean_variance --frontier
python -m portfolio_builder.optimization --age 55 --risk 4 --method mean_variance --objective max_sharpe
```

Unconstrained mean-variance results depend heavily on the historical sample. Since 2012, SPY has had the best return and Sharpe ratio, so the frontier is mostly SPY blended with short-term TIPS. Use `max_weight` (or `--lookback-years`) for more diversified portfolios.

## Service layer

`portfolio_builder/service` connects a UI to the optimization engine. It checks the client's form inputs, runs both approaches on the cached market data, and returns one response ready to display.

| Input | Range | Effect |
|---|---|---|
| `risk_tolerance` | 1–10, or `conservative` (3) / `moderate` (5.5) / `aggressive` (8) | Sets the equity vs. bond mix |
| `horizon_years` | 1–30 | Adjusts risk: under 3 years capped at 3; 3–4 years −2; 5–9 years −1; 10–19 years no change; 20+ years +1 |
| `initial_investment` | $1,000–$10,000,000 | Converted to a dollar amount for each holding |
| `monthly_contribution` | $0–$50,000 | Included in the projection |
| `goal` | retirement / home purchase / education / general wealth | Adds goal-specific notes |
| `age` | 18–80 | Used in the rule-based lifecycle allocation |

```python
from portfolio_builder.service import InputValidationError, get_form_options, recommend_portfolio

options = get_form_options()          # label, help, widget, min/max/step, default, choices for each field

try:
    response = recommend_portfolio(risk_tolerance="moderate", horizon_years=20, initial_investment=100_000,
                                   monthly_contribution=1_000, goal="retirement", age=45)
except InputValidationError as exc:
    exc.field_errors                  # {"age": "Age must be a number between 18 and 80", ...}

response.profile                      # entered vs. effective risk tolerance, risk band, horizon adjustment
response.rule_based.holdings          # ticker, fund name, asset class, role, weight, $ amount
response.mean_variance.frontier_position   # return gap vs. the frontier, position 0..1, on_frontier
response.efficient_frontier           # frontier points, reference portfolios, client and rule-based points
response.notes                        # goal and horizon context
response.holdings_frame("mean_variance")   # DataFrames for tables and charts:
response.frontier_frame()             #   also asset_class_frame, projection_frame, comparison_frame
response.to_dict()                    # JSON-safe dict
```

Projections show value at each year end, with each month's contribution added after that month's growth. They include an expected path plus 10th, 25th, 50th, 75th and 90th percentiles from 5,000 lognormal simulations with a fixed seed, so repeated runs give the same result. The response also includes `probability_of_meeting_target` against `target_amount` (default set by goal), a 10–20 year `backtest` against the S&P 500 (`backtest_years`, `rebalance`), asset-class risk/return points, and non-blocking `warnings`. The service loads the engine once per process and is safe to call from concurrent requests.

```bash
python -m portfolio_builder.service --age 45 --risk moderate --horizon 20 --initial 100000 --monthly 1000 --goal retirement
python -m portfolio_builder.service --age 30 --risk aggressive --horizon 2 --goal home_purchase --json
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest              # offline tests (fake connector)
pytest -m network   # live Yahoo Finance download test
```

## Deployments

- GitHub: https://github.com/hewness/FIN.6515.21.FA2026.WK3.AI_robo_advisor_portfolio_builder_app
- Hugging Face Space: https://huggingface.co/spaces/hewness/ai_robo_advisor_portfolio_builder_app
