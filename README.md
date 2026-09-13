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

## Tests

```bash
pip install -r requirements-dev.txt
pytest              # offline tests (fake connector)
pytest -m network   # live Yahoo Finance download test
```

## Deployments

- GitHub: https://github.com/hewness/FIN.6515.21.FA2026.WK3.AI_robo_advisor_portfolio_builder_app
- Hugging Face Space: https://huggingface.co/spaces/hewness/ai_robo_advisor_portfolio_builder_app
