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

## Tests

```bash
pip install -r requirements-dev.txt
pytest              # offline tests (fake connector)
pytest -m network   # live Yahoo Finance download test
```

## Deployments

- GitHub: https://github.com/hewness/FIN.6515.21.FA2026.WK3.AI_robo_advisor_portfolio_builder_app
- Hugging Face Space: https://huggingface.co/spaces/hewness/ai_robo_advisor_portfolio_builder_app
