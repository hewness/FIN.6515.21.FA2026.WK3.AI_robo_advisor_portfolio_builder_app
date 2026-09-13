import pandas as pd
import pytest

from portfolio_builder.market_data import HISTORY_COLUMNS, MarketDataError, YFinanceConnector
from portfolio_builder.market_data.connectors import yfinance_connector as yc


def raw_yahoo_frame(with_capital_gains: bool = True) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        ["2024-01-03 00:00:00", "2024-01-02 00:00:00", "2024-01-03 00:00:00"], name="Date"
    ).tz_localize("America/New_York")
    data = {
        "Open": [1.0, 2.0, 3.0],
        "High": [1.0, 2.0, 3.0],
        "Low": [1.0, 2.0, 3.0],
        "Close": [1.0, 2.0, 3.0],
        "Adj Close": [0.9, 1.9, 2.9],
        "Volume": [10, 20, 30],
        "Dividends": [0.0, 0.1, 0.0],
        "Stock Splits": [0.0, 0.0, 0.0],
    }
    if with_capital_gains:
        data["Capital Gains"] = [0.0, 0.0, 0.0]
    return pd.DataFrame(data, index=idx)


class FakeTicker:
    responses: list = []
    calls: list = []

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        FakeTicker.calls.append((self.symbol, kwargs))
        result = FakeTicker.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def info(self):
        return {"longName": "Fund", "netExpenseRatio": 0.03, "irrelevant": object(), "yield": None}


@pytest.fixture
def fake_yf(monkeypatch):
    FakeTicker.responses = []
    FakeTicker.calls = []
    monkeypatch.setattr(yc.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yc.time, "sleep", lambda s: None)
    return FakeTicker


def test_normalizes_yahoo_frame(fake_yf):
    fake_yf.responses = [raw_yahoo_frame(with_capital_gains=False)]
    df = YFinanceConnector().fetch_history("SPY")

    assert list(df.columns) == list(HISTORY_COLUMNS)
    assert df.index.tz is None and df.index.name == "date"
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df.loc["2024-01-03", "close"] == 3.0  # duplicate date keeps last row
    assert (df["capital_gains"] == 0).all()
    assert fake_yf.calls[0][1] == {"auto_adjust": False, "actions": True, "period": "max"}


def test_retries_then_succeeds(fake_yf):
    fake_yf.responses = [RuntimeError("rate limited"), pd.DataFrame(), raw_yahoo_frame()]
    df = YFinanceConnector(max_retries=3).fetch_history("VTI")
    assert len(df) == 2
    assert len(fake_yf.calls) == 3


def test_gives_up_with_market_data_error(fake_yf):
    fake_yf.responses = [pd.DataFrame(), pd.DataFrame()]
    with pytest.raises(MarketDataError, match="VTIP"):
        YFinanceConnector(max_retries=2).fetch_history("VTIP")


def test_info_keeps_known_non_null_fields(fake_yf):
    assert YFinanceConnector().fetch_info("SPY") == {"longName": "Fund", "netExpenseRatio": 0.03}


@pytest.mark.network
def test_live_download_spy():
    df = YFinanceConnector().fetch_history("SPY")
    assert df.index.min() < pd.Timestamp("1994-01-01")
    assert list(df.columns) == list(HISTORY_COLUMNS)
