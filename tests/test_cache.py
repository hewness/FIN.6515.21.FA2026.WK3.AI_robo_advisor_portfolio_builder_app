import numpy as np
import pandas as pd
import pytest

from portfolio_builder.market_data import HISTORY_COLUMNS
from portfolio_builder.market_data.connectors import validate_history
from portfolio_builder.market_data.cache import normalize_ticker

from .conftest import make_history


def test_round_trip_preserves_schema(cache):
    df = make_history()
    cache.write_history("spy", df, source="fake", source_version="1")

    out = cache.read_history("SPY")
    assert list(out.columns) == list(HISTORY_COLUMNS)
    assert out.index.name == "date"
    assert out["volume"].dtype == "int64"
    assert out["adj_close"].dtype == "float64"
    assert out.index.dtype == "datetime64[ns]"
    pd.testing.assert_frame_equal(out, validate_history(df, "SPY"), check_freq=False)
    np.testing.assert_allclose(out["adj_close"], df["adj_close"])
    assert not list(cache.prices_dir.glob("*.tmp"))


def test_manifest_and_status(cache):
    df = make_history(start="2021-03-01", periods=10)
    cache.write_history("VTI", df, source="fake")

    entry = cache.manifest()["VTI"]
    assert entry["first_date"] == "2021-03-01"
    assert entry["rows"] == 10
    assert entry["source"] == "fake"

    status = cache.status(["VTI", "BND"])
    assert bool(status.loc["VTI", "cached"]) is True
    assert bool(status.loc["BND", "cached"]) is False


def test_info_round_trip_and_delete(cache):
    cache.write_history("AGG", make_history(periods=5))
    cache.write_info("AGG", {"longName": "iShares Core US Aggregate Bond ETF"})
    assert cache.read_info("agg")["longName"].startswith("iShares")

    cache.delete("AGG")
    assert not cache.has_history("AGG") and not cache.has_info("AGG")
    assert "AGG" not in cache.manifest()


def test_read_missing_raises(cache):
    with pytest.raises(FileNotFoundError):
        cache.read_history("SPY")


@pytest.mark.parametrize("bad", ["../etc", "SP Y", "", "..", "a/b", "C:\\x"])
def test_invalid_tickers_rejected(bad):
    with pytest.raises(ValueError):
        normalize_ticker(bad)


def test_index_tickers_allowed():
    assert normalize_ticker("^irx") == "^IRX"
    assert normalize_ticker("brk.b") == "BRK.B"
