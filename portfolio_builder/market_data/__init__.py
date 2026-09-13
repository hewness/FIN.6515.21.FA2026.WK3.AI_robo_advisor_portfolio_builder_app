"""Market data connector and local cache.

Typical use::

    from portfolio_builder.market_data import get_market_data_service

    svc = get_market_data_service()
    returns = svc.get_returns(frequency="monthly")   # cached on disk after first call
    svc.refresh(["SPY"])                             # force a re-download
"""

from .cache import ParquetCache
from .connectors import HISTORY_COLUMNS, MarketDataConnector, MarketDataError, YFinanceConnector
from .service import MarketDataService, get_market_data_service

__all__ = [
    "HISTORY_COLUMNS",
    "MarketDataConnector",
    "MarketDataError",
    "MarketDataService",
    "ParquetCache",
    "YFinanceConnector",
    "get_market_data_service",
]
