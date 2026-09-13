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

`app.py` launches a Gradio dashboard (`portfolio_builder/ui`) that compares three portfolios side by side: rule-based, mean-variance and research-informed.

**Sidebar inputs** (defaults in brackets). The dashboard updates when you release a slider, change a choice, or leave a text field. Money fields format as you type (`$#,000`) and accept shorthand such as `50k` or `1.2m`. Each label sits beside its input, with all labels in one column so the inputs line up; sliders show their number box beside the label and the track below. Hover over a label's ⓘ to see what the input does and its allowed range.

| Group | Input | Control | Range / options | Default |
|---|---|---|---|---|
| About you | Age | Number | 18–80 | 40 |
| | Financial goal | Dropdown | Retirement / Home purchase / Education / General wealth | Retirement |
| | Goal target | Money text | $1,000–$100M; resets to the goal's default when the goal changes | $1.5M (home $150k, education $200k, general $1M) |
| Income | Annual income | Money text | $0–$5,000,000 ($0 if retired) | $85,000 |
| | Retirement income | Money text | $0–$1,000,000 per year of Social Security and pensions; blank = 40% of annual income | blank (auto) |
| | Retirement age | Number | 50–75 | 67 |
| Risk profile | Hump-shaped equity glide path | Checkbox | On / Off | On |
| | Risk tolerance | Dropdown (Conservative / Moderate / Aggressive / Custom) + 1–10 slider | presets = 3 / 5.5 / 8 | Moderate (5.5) |
| Investment plan | Initial investment | Money text | $1,000–$10,000,000 | $50,000 |
| | Monthly contribution | Money text | $0–$50,000 | $1,000 |
| | Investment horizon (yrs) | Slider | 1–30 years | 25 |
| Backtest settings | Lookback (yrs) | Slider | 10–20 years | 10 |
| | Rebalancing | Dropdown | Monthly / Quarterly / Annual | Quarterly |

Invalid inputs are listed in a red status box in the sidebar, and the charts keep the last valid results. Input combinations worth a second look (an unreachable target, a horizon beyond age 75, a very long home or education horizon, a capped short horizon, $0 income before retirement age) show an amber warning.

**Main panel**
- **A card for each portfolio** (Rule-based Lifecycle Portfolio in indigo, Mean-variance Optimized Portfolio in teal, Research-informed Portfolio in violet), three across; on narrow screens they stack. Inside each card the panels stack:
  - A one-line **equity basis**: how that card's stock share was set (glide path, equity band, or the practical finance formula with its numbers). When the research formula asks for more than 100% stocks (large human capital relative to savings), the line turns amber ("⚠ Capped at 100% stocks · formula gives 520%") and the card's description explains that risk, age and horizon changes won't move the mix until savings grow.
  - Tiles (three over two) for expected annual return, volatility, Sharpe ratio, maximum historical drawdown (from the backtest), and probability of reaching the goal (share of 5,000 simulated outcomes at or above the target).
  - An allocation-by-asset-class donut above the holdings table. The table reserves room for every holding (up to 6 funds plus cash), so it never scrolls, and each section lines up across the three cards. Each holding row starts with a color swatch that matches its asset class's slice, and shows the fund's annualized expected return (Exp. ret., the same estimate the optimizer uses; weights times these returns give the portfolio's expected return), its weight and its dollar amount. Hovering over (or tapping) a slice highlights that asset class's holdings and dims the rest.
  - Projected wealth: expected (mean), optimistic (75th percentile) and pessimistic (25th percentile) paths, including contributions, with the goal target line. All cards use the same scale.
- **Research vs. Popular Wisdom:** bars comparing the stock share of the three portfolios with two popular rules of thumb (100 − age, and a typical target-date fund), next to a short explanation of where the research-informed model disagrees with popular advice and why, citing the papers. See [Research-informed model](#research-informed-model).
- **Rule-based vs Mean-variance vs Research-informed:** a comparison container with two charts side by side that each plot all three portfolios.
  - **Risk vs. Return:** asset classes (each an equal-weight blend of its funds), cash, the efficient frontier, the max-Sharpe point and a star for each portfolio.
  - **Historical Backtest vs. S&P 500 (SPY):** growth of the initial investment and drawdowns over 10–20 years.
- **Notes & assumptions**, including the full name of each fund.

**Colors:**
- Indigo, teal, violet and slate always mean the rule-based portfolio, the mean-variance portfolio, the research-informed portfolio and the S&P 500 benchmark, in every widget. Popular rules of thumb are neutral gray.
- Asset classes use their own palette so they never look like a portfolio: warm tones for stocks, greens for bonds, warm gray for cash.
- The efficient frontier is a dashed sky-blue line, a color not used by any portfolio or asset class.

**Backtest method** (`portfolio_builder/backtest`):
- Daily total returns with fixed target weights. Holdings drift and reset at each rebalance date. Cash earns the risk-free rate (0% by default).
- Before a fund existed, an older ETF tracking the same asset class stands in: VTI→SPY, VXUS→EFA, VWO→EEM, BND→AGG, VTIP→TIP (`HISTORY_PROXIES` in `universe.py`). These history proxies are used only to extend the backtest; they are never recommended or held, and the backtest notes list where they were used.
- It models the initial investment only, with no contributions, fees or taxes.
- Mean-variance weights are estimated on overlapping history, so that backtest is in-sample.

## Equity glide path

The **Hump-shaped equity glide path** toggle (on by default) controls how equity changes with age in the rule-based and mean-variance models. The research-informed model does not use it; its age effect comes from human capital (below).

| Age | 18–25 | 35 | 45 | 55 | 65+ |
|---|---|---|---|---|---|
| Hump-shaped base equity | 60% | 70% | 80% | 70% | 60% |
| Linear base equity (110 − age, toggle off) | 92–85% | 75% | 65% | 55% | ≤45% |

- The hump path is piecewise-linear through 60% at 25, 80% at 45 and 60% at 65, and flat outside those ages (`HumpGlidePath` in `portfolio_builder/optimization/glide_path.py`).
- Risk tolerance shifts the base by −20 points (risk 1) to +20 points (risk 10), and the result is kept between 10% and 100%. This is the same shift as the linear rule.
- **Rule-based:** the equity share above sets the equity sleeve (US large-cap, international developed, emerging markets, REITs); bonds, TIPS and cash fill the rest.
- **Mean-variance:** equity funds (VTI, VXUS, VWO, VNQ) must total that equity share ±5 points; BND and VTIP fill the rest. Risk tolerance still chooses the volatility target, and the efficient frontier, reference portfolios and client portfolio are all solved within the band. The risk/return chart labels this frontier "glide-path equity band".
- With the toggle off, both models run exactly as before: the linear rule and an unconstrained mean-variance optimization.

## Research-informed model

The third model (`research_informed`, `portfolio_builder/optimization/research_informed.py`) sets the equity share from **wealth and income, not just age**, using the practical finance approximation of Choi, Liu & Liu (2025), *Practical Finance: An Approximate Solution to Lifecycle Portfolio Choice* (equation 9):

```
equity = clip( Merton share × (1 + human capital ÷ financial wealth), 0%, 100% )
Merton share = (log equity premium + σ²/2) ÷ (γ · σ²)
```

- **Financial wealth** is the initial investment.
- **Human capital** is the present value of future income through age 100: annual income until the retirement age, then retirement income (Social Security and pensions, 40% of income by default). Human capital behaves like a bond, so the more of it you have relative to savings, the more stock the portfolio can hold.
- **Discounting:** each year's income is discounted with the paper's age-varying one-year-ahead rates. Working-life rates use Table 1, column 3: `0.087·γ/10 − 0.267·π + 1.132·r + 4.332·0.130² + 0.028·0.242² + 0.010·d − 0.149·age/100 + 0.142·(age/100)² − 0.020`, with college-graduate income risk and replacement rate d. Retirement rates use Table 2, column 3: `0.0003·γ/10 − 0.217·π + 0.893·r + 0.476·age/100 − 0.295·(age/100)² − 0.166`. The implementation reproduces the paper's Table 3 example exactly (human capital $924,805, 30% equity).
- **Assumptions:** these are long-run research values (the paper's baseline), not the app's historical estimates. Log equity premium π = 4%, stock volatility σ = 18.5%, log real risk-free rate r = 2%.
- **Risk tolerance** (horizon-adjusted, like the other models) maps linearly onto the paper's relative risk aversion grid: γ = 10 at risk 1, down to γ = 4 at risk 10. Moderate with a 20+ year horizon (6.5) gives γ ≈ 6.3 and a Merton share of 26%.
- **Funds:** equity goes 45% US / 35% international developed / 12% emerging / 8% REITs. That is about half non-US, closer to global market-cap weights (Choi 2022 notes popular books hold only ~27% abroad). The rest goes 65% aggregate bonds / 30% TIPS / 5% cash.

| Client (Moderate, 20-year horizon) | Human capital | Research-informed equity | 100 − age | Typical target-date fund |
|---|---|---|---|---|
| 68, retired, $500k invested, $30k/yr Social Security + pension | $584k | **57%** | 32% | 39% |
| 68, retired, $1M invested, $30k/yr | $584k | 42% | 32% | 39% |
| 40, $85k income, $50k invested | $937k | 100% (capped) | 60% | 90% |

**Where it disagrees with popular wisdom, and why.** The dashboard shows these points for the current client:
- **Human capital.** Popular rules look only at age, and only 8 of 47 best-selling finance books mention human capital (Choi 2022, *Popular Personal Financial Advice versus the Professors*). The research model counts future earnings and pensions as a bond-like asset.
- **Retirees keep more stock.** Duarte, Fonseca, Goodman & Parker (2022, NBER WP 29559) find optimal equity stays near 60% at and through retirement: retirees still have long horizons and pension-like income. Target-date funds glide down to 30–40%, which costs about 2–3% of consumption a year. The typical 68-year-old retiree above lands at 57%, not 30–40%.
- **Young savers with little wealth** can hold up to 100% stocks. Many popular books warn against this or keep near-term money in cash.
- **Wealthy investors** hold less stock than age rules suggest, because their savings must carry the risk alone.
- **Time diversification.** Popular books say stocks get safer over time. In the research model, equity falls with age only because remaining human capital shrinks.

```python
from portfolio_builder.optimization import InvestorProfile, get_optimization_engine

profile = InvestorProfile(age=68, risk_tolerance=6.5, annual_income=0, retirement_income=30_000,
                          financial_wealth=500_000)
result = get_optimization_engine().optimize("research_informed", profile)
result.details["equity_pct"], result.details["human_capital"], result.details["merton_share"]  # 0.571, 583860, 0.263
```

## Investment universe

| Asset class | Role in portfolio | Tickers |
|---|---|---|
| US large-cap stocks | Growth, domestic equity exposure | VTI |
| International developed stocks | Diversification, international exposure | VXUS |
| Emerging market stocks | Higher growth potential, higher risk | VWO |
| US Aggregate bonds | Stability, income | BND |
| Treasury inflation-protected securities | Inflation Hedge | VTIP |
| Real estate (REITs) | Real asset diversification | VNQ |
| Cash and Money Markets | Liquidity, capital preservation | *(no ticker)* |

One fund per asset class, defined in `portfolio_builder/universe.py`. SPY, EFA, EEM, AGG and TIP are cached as backtest-only history proxies (see *Backtest method*); SPY is also the S&P 500 benchmark.

## Market data

Daily history comes from Yahoo Finance (`yfinance`) and is cached on disk under `data/market_data/`:

- `prices/<TICKER>.parquet`: full history with `open, high, low, close, adj_close, volume, dividends, stock_splits, capital_gains`
- `info/<TICKER>.json`: fund name, category, expense ratio (in percent) and similar fields
- `manifest.json`: when each ticker was downloaded and the date range it covers

A snapshot of the cache is committed to the repo (the Parquet files are stored with Git LFS). Data is read from disk when it's there and downloaded when it isn't. Set `MARKET_DATA_CACHE_DIR` to use a different folder.

### Command line

```bash
python -m portfolio_builder.market_data status                  # what is cached
python -m portfolio_builder.market_data download                # download missing tickers (universe + history proxies)
python -m portfolio_builder.market_data refresh                 # force re-download of everything
python -m portfolio_builder.market_data refresh --tickers VTI   # force re-download of one ticker
```

### Python

```python
from portfolio_builder.market_data import get_market_data_service

svc = get_market_data_service()
prices  = svc.get_prices()                          # adjusted close, dates x tickers, common dates only
returns = svc.get_returns(frequency="monthly")      # total returns; also "daily"/"weekly", method="log"
cov     = returns.cov() * 12                        # annualized covariance for optimization
divs    = svc.get_dividends(["BND", "VTIP"])        # cash dividends per share
meta    = svc.get_universe_metadata()               # asset class, role, name, expense ratio
svc.refresh(["VTI"])                                # re-download; keeps old data if the download fails
```

Every method defaults to the full universe. Use `align="outer"` to keep each fund's full history (older funds start before VTIP's 2012 inception). A data source other than yfinance can be plugged in by implementing `MarketDataConnector` (`portfolio_builder/market_data/connectors/base.py`).

## Portfolio optimization

`portfolio_builder/optimization` turns a client profile into a portfolio. A profile is an age plus a **risk tolerance from 1 (most conservative) to 10 (most aggressive)**, with optional income, retirement income, retirement age and financial wealth (used only by the research-informed model). Expected returns and covariance are estimated from the cached monthly total returns over the window where all funds have data, then annualized. Every portfolio is long-only (no shorting) and its weights sum to 1; results that break either rule are rejected.

| Approach | How it allocates |
|---|---|
| `rule_based` | Equity % = 110 − age, shifted from −20 points (risk 1) to +20 points (risk 10) and kept between 10% and 100%. Equity is split 55% US large-cap / 25% international developed / 10% emerging / 10% REITs. The rest is split 70% aggregate bonds / 20% TIPS / 10% cash. Each asset class's weight goes to its single fund. (If a class were given several funds, they would be ranked by historical volatility: conservative clients get the calmer fund, aggressive clients the more volatile one, and clients in between a blend.) |
| `research_informed` | Equity % = Merton share × (1 + human capital ÷ financial wealth), clipped to 0–100% (see [Research-informed model](#research-informed-model)). Split 45/35/12/8 across US, international, emerging and REITs, and 65/30/5 across bonds, TIPS and cash. Requires `financial_wealth`. |
| `mean_variance` | Builds the long-only efficient frontier with `scipy.optimize.minimize` (SLSQP). The client portfolio has the highest return for a target volatility, placed by risk tolerance between the minimum-volatility portfolio (risk 1) and the maximum-return portfolio (risk 10). Other objectives: `max_sharpe`, `min_volatility`, `target_volatility`, `target_return`. The result also includes the frontier and the min-volatility, max-Sharpe and max-return reference portfolios. |

Cash (shown as `CASH`) earns the risk-free rate, which defaults to **0%**, at zero volatility. The same rate is the Sharpe ratio's risk-free rate, so Sharpe = expected return ÷ volatility. Only the rule-based and research-informed approaches hold cash. Pass `risk_free_rate=` to the engine to change it.

```python
from portfolio_builder.optimization import InvestorProfile, get_optimization_engine

engine = get_optimization_engine()                    # risk_free_rate defaults to 0.0
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
python -m portfolio_builder.optimization --age 40 --risk 6                       # all approaches
python -m portfolio_builder.optimization --age 68 --risk 6.5 --method research_informed --wealth 500000 --retirement-income 30000
python -m portfolio_builder.optimization --age 30 --risk 8 --method mean_variance --frontier
python -m portfolio_builder.optimization --age 55 --risk 4 --method mean_variance --objective max_sharpe
```

Unconstrained mean-variance results depend heavily on the historical sample. Since 2012, VTI has had the best return and Sharpe ratio, so the frontier is mostly VTI blended with short-term TIPS (VTIP). Use `max_weight` (or `--lookback-years`) for more diversified portfolios.

## Service layer

`portfolio_builder/service` connects a UI to the optimization engine. It checks the client's form inputs, runs all three approaches on the cached market data, and returns one response ready to display.

| Input | Range | Effect |
|---|---|---|
| `risk_tolerance` | 1–10, or `conservative` (3) / `moderate` (5.5) / `aggressive` (8) | Sets the equity vs. bond mix |
| `horizon_years` | 1–30 | Adjusts risk: under 3 years capped at 3; 3–4 years −2; 5–9 years −1; 10–19 years no change; 20+ years +1 |
| `initial_investment` | $1,000–$10,000,000 | Converted to a dollar amount for each holding |
| `monthly_contribution` | $0–$50,000 | Included in the projection |
| `goal` | retirement / home purchase / education / general wealth | Adds goal-specific notes |
| `age` | 18–80 | Used in the lifecycle glide path and to value future income |
| `annual_income` | $0–$5,000,000 (default $85,000) | Research-informed human capital (ignored once age ≥ retirement age) |
| `retirement_income` | $0–$1,000,000 per year, or blank for 40% of income | Research-informed human capital in retirement |
| `retirement_age` | 50–75 (default 67) | When retirement income replaces earnings |

```python
from portfolio_builder.service import InputValidationError, get_form_options, recommend_portfolio

options = get_form_options()          # label, help, widget, min/max/step, default, choices for each field

try:
    response = recommend_portfolio(risk_tolerance="moderate", horizon_years=20, initial_investment=100_000,
                                   monthly_contribution=1_000, goal="retirement", age=45)
except InputValidationError as exc:
    exc.field_errors                  # {"age": "Age must be a number between 18 and 80", ...}

response.profile                      # entered vs. effective risk tolerance, risk band, horizon adjustment
response.rule_based.holdings          # ticker, fund name, asset class, role, weight, $ amount, expected return
response.mean_variance.frontier_position   # return gap vs. the frontier, position 0..1, on_frontier
response.research_informed.details    # equity_pct, merton_share, human_capital, risk_aversion, ...
response.research_insight             # equity comparisons vs. popular rules, headline, points, sources
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
python -m portfolio_builder.service --age 68 --risk moderate --horizon 20 --initial 500000 --income 0 --retirement-income 30000
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
