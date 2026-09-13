"""Historical backtest of fixed-weight portfolios with periodic rebalancing against a benchmark."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..market_data import MarketDataService
from ..optimization.models import CASH
from ..universe import get_history_proxies

Rebalance = Literal["monthly", "quarterly", "annual"]
_PERIOD_CODES = {"monthly": "M", "quarterly": "Q", "annual": "Y"}
TRADING_DAYS = 252
BENCHMARK_KEY = "benchmark"


class BacktestError(Exception):
    """Raised when a backtest cannot be run (e.g. missing price history)."""


@dataclass(frozen=True)
class BacktestConfig:
    years: int = 15
    rebalance: Rebalance = "quarterly"
    benchmark: str = "SPY"
    risk_free_rate: float = 0.04
    use_proxies: bool = True

    def __post_init__(self) -> None:
        if self.years < 1:
            raise ValueError("years must be at least 1")
        if self.rebalance not in _PERIOD_CODES:
            raise ValueError(f"rebalance must be one of {sorted(_PERIOD_CODES)}")


@dataclass(frozen=True)
class BacktestMetrics:
    final_value: float
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_date: str | None
    best_year: float | None
    worst_year: float | None


@dataclass
class BacktestResult:
    values: pd.DataFrame      # daily value per portfolio name and "benchmark"
    drawdowns: pd.DataFrame   # daily drawdown (<= 0) for the same columns
    metrics: dict[str, BacktestMetrics]
    start: pd.Timestamp
    end: pd.Timestamp
    years_requested: int
    rebalance: str
    benchmark: str
    initial_value: float
    proxies_used: dict[str, dict[str, str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def years_covered(self) -> float:
        return (self.end - self.start).days / 365.25


class Backtester:
    """Runs backtests from cached market data; daily return panels are memoized per ticker set."""

    def __init__(self, market_data: MarketDataService) -> None:
        self.market_data = market_data
        self._returns_cache: dict[tuple[str, ...], pd.DataFrame] = {}
        self._lock = threading.Lock()

    def clear_cache(self) -> None:
        with self._lock:
            self._returns_cache.clear()

    def proxy_map(self, tickers: list[str] | None = None) -> dict[str, str]:
        """History proxy (older same-asset-class ETF) for each fund in ``tickers``; see ``HISTORY_PROXIES``."""
        proxies = get_history_proxies()
        return {t: p for t, p in proxies.items() if tickers is None or t in tickers}

    def run(
        self,
        portfolios: dict[str, pd.Series],
        config: BacktestConfig | None = None,
        initial_value: float = 10_000.0,
    ) -> BacktestResult:
        config = config or BacktestConfig()
        if not portfolios:
            raise BacktestError("No portfolios to backtest")
        if BENCHMARK_KEY in portfolios:
            raise BacktestError(f"'{BENCHMARK_KEY}' is reserved for the benchmark series")
        weights = {name: _clean(w) for name, w in portfolios.items()}

        risky = sorted({t for w in weights.values() for t in w.index if t != CASH})
        proxies = self.proxy_map(risky) if config.use_proxies else {}
        needed = sorted(set(risky) | {config.benchmark} | {proxies[t] for t in risky if t in proxies})
        raw = self._daily_returns(needed)

        filled = raw.copy()
        first_valid: dict[str, pd.Timestamp] = {}
        for t in needed:
            idx = raw[t].first_valid_index()
            if idx is None:
                raise BacktestError(f"No price history for {t}")
            first_valid[t] = idx
        for t, proxy in proxies.items():
            before = filled.index < first_valid[t]
            filled.loc[before, t] = raw.loc[before, proxy]

        used = sorted(set(risky) | {config.benchmark})
        end = min(raw[t].last_valid_index() for t in used)
        target_start = end - pd.DateOffset(years=config.years)
        # A series' first return is on the day after its first price, so the window can start on that price day.
        first_returns = [filled.index.get_loc(filled[t].first_valid_index()) for t in used]
        earliest = filled.index[max(max(first_returns) - 1, 0)]
        # The window starts on a trading day; returns are applied from the next day on.
        start = filled.index[filled.index >= max(target_start, earliest)][0]
        notes: list[str] = []
        if start > target_start + pd.Timedelta(days=7):
            notes.append(
                f"Only {(end - start).days / 365.25:.1f} years of history are available; "
                f"the backtest starts {start.date().isoformat()} instead of {config.years} years back."
            )

        window = filled.loc[(filled.index > start) & (filled.index <= end), used]
        if window.isna().any().any():
            notes.append("Missing daily returns inside the window were treated as 0%.")
            window = window.fillna(0.0)

        proxies_used = {
            t: {"proxy": p, "until": first_valid[t].date().isoformat()}
            for t, p in proxies.items()
            if first_valid[t] > start
        }
        if proxies_used:
            listed = ", ".join(f"{t} (via {v['proxy']} until {v['until']})" for t, v in sorted(proxies_used.items()))
            notes.append(
                "Before these funds existed, an older ETF tracking the same asset class stood in "
                f"(history only, never held): {listed}."
            )

        values = pd.DataFrame(index=pd.DatetimeIndex([start, *window.index], name="date"))
        for name, w in weights.items():
            values[name] = _simulate(w, window, config, initial_value)
        values[BENCHMARK_KEY] = initial_value * np.concatenate(
            [[1.0], np.cumprod(1.0 + window[config.benchmark].to_numpy())]
        )

        drawdowns = values / values.cummax() - 1.0
        metrics = {col: _metrics(values[col], drawdowns[col], config.risk_free_rate) for col in values.columns}
        return BacktestResult(
            values=values,
            drawdowns=drawdowns,
            metrics=metrics,
            start=start,
            end=end,
            years_requested=config.years,
            rebalance=config.rebalance,
            benchmark=config.benchmark,
            initial_value=initial_value,
            proxies_used=proxies_used,
            notes=notes,
        )

    def _daily_returns(self, tickers: list[str]) -> pd.DataFrame:
        key = tuple(tickers)
        with self._lock:
            cached = self._returns_cache.get(key)
        if cached is None:
            prices = self.market_data.get_prices(tickers, field="adj_close", align="outer")
            cached = prices.pct_change(fill_method=None)
            with self._lock:
                self._returns_cache[key] = cached
        return cached


def run_backtest(
    portfolios: dict[str, pd.Series],
    market_data: MarketDataService,
    config: BacktestConfig | None = None,
    initial_value: float = 10_000.0,
) -> BacktestResult:
    return Backtester(market_data).run(portfolios, config, initial_value)


def _clean(weights: pd.Series) -> pd.Series:
    w = pd.Series(weights, dtype="float64")
    w = w[w > 0]
    if w.empty or abs(w.sum() - 1.0) > 1e-6:
        raise BacktestError("Portfolio weights must be positive and sum to 1")
    return w / w.sum()


def _simulate(weights: pd.Series, window: pd.DataFrame, config: BacktestConfig, initial_value: float) -> np.ndarray:
    """Fixed target weights, drifting within each period and reset at the start of the next."""
    cash_w = float(weights.get(CASH, 0.0))
    risky = weights.drop(CASH, errors="ignore")
    returns = window[risky.index].to_numpy()
    w = risky.to_numpy()
    cash_daily = (1.0 + config.risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    periods = window.index.to_period(_PERIOD_CODES[config.rebalance])
    boundaries = np.flatnonzero(np.r_[True, periods[1:] != periods[:-1]])

    out = np.empty(len(window) + 1)
    out[0] = value = initial_value
    for i, lo in enumerate(boundaries):
        hi = boundaries[i + 1] if i + 1 < len(boundaries) else len(window)
        growth = np.cumprod(1.0 + returns[lo:hi], axis=0)
        cash_growth = (1.0 + cash_daily) ** np.arange(1, hi - lo + 1)
        segment = value * (growth @ w + cash_w * cash_growth)
        out[lo + 1 : hi + 1] = segment
        value = segment[-1]
    return out


def _metrics(values: pd.Series, drawdown: pd.Series, risk_free_rate: float) -> BacktestMetrics:
    daily = values.pct_change().dropna()
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1e-9)
    total = values.iloc[-1] / values.iloc[0] - 1.0
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0
    vol = float(daily.std() * np.sqrt(TRADING_DAYS)) if len(daily) > 1 else 0.0
    sharpe = (float(daily.mean()) * TRADING_DAYS - risk_free_rate) / vol if vol > 0 else 0.0
    max_dd = float(drawdown.min())

    year_end = values.groupby(values.index.year).last()
    yearly = year_end.pct_change().dropna()
    if not yearly.empty and values.index[-1] < pd.Timestamp(year=values.index[-1].year, month=12, day=20):
        yearly = yearly.iloc[:-1]  # drop the partial final year
    return BacktestMetrics(
        final_value=float(values.iloc[-1]),
        total_return=float(total),
        cagr=float(cagr),
        volatility=vol,
        sharpe_ratio=float(sharpe),
        max_drawdown=max_dd,
        max_drawdown_date=None if max_dd == 0 else drawdown.idxmin().date().isoformat(),
        best_year=None if yearly.empty else float(yearly.max()),
        worst_year=None if yearly.empty else float(yearly.min()),
    )
