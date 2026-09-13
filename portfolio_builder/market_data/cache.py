"""Local on-disk cache for market data: one Parquet file per ticker plus a JSON manifest."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .connectors.base import validate_history

_TICKER_RE = re.compile(r"^[A-Z0-9.^-]{1,20}$")

STATUS_COLUMNS = ["first_date", "last_date", "rows", "downloaded_at", "source", "source_version"]


def normalize_ticker(ticker: str) -> str:
    """Upper-case and validate a ticker so it is safe to use as a file name."""
    t = str(ticker).strip().upper()
    if not _TICKER_RE.match(t) or t.strip(".") == "":
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return t


class ParquetCache:
    """Layout under ``root``::

        prices/<TICKER>.parquet   daily history in the normalized schema
        info/<TICKER>.json        descriptive fields (name, expense ratio, ...)
        manifest.json             per-ticker download metadata
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.prices_dir = self.root / "prices"
        self.info_dir = self.root / "info"
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.Lock()

    # -- history -----------------------------------------------------------
    def history_path(self, ticker: str) -> Path:
        return self.prices_dir / f"{normalize_ticker(ticker)}.parquet"

    def has_history(self, ticker: str) -> bool:
        return self.history_path(ticker).is_file()

    def read_history(self, ticker: str) -> pd.DataFrame:
        path = self.history_path(ticker)
        if not path.is_file():
            raise FileNotFoundError(f"No cached history for {normalize_ticker(ticker)} at {path}")
        return pd.read_parquet(path)

    def write_history(
        self,
        ticker: str,
        df: pd.DataFrame,
        source: str = "unknown",
        source_version: str | None = None,
    ) -> None:
        t = normalize_ticker(ticker)
        df = validate_history(df, t)
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        path = self.history_path(t)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, engine="pyarrow")
        os.replace(tmp, path)
        self._update_manifest(
            t,
            {
                "first_date": df.index.min().date().isoformat(),
                "last_date": df.index.max().date().isoformat(),
                "rows": int(len(df)),
                "downloaded_at": _utc_now(),
                "source": source,
                "source_version": source_version,
            },
        )

    # -- info --------------------------------------------------------------
    def info_path(self, ticker: str) -> Path:
        return self.info_dir / f"{normalize_ticker(ticker)}.json"

    def has_info(self, ticker: str) -> bool:
        return self.info_path(ticker).is_file()

    def read_info(self, ticker: str) -> dict[str, Any]:
        path = self.info_path(ticker)
        if not path.is_file():
            raise FileNotFoundError(f"No cached info for {normalize_ticker(ticker)} at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_info(self, ticker: str, info: dict[str, Any]) -> None:
        self.info_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.info_path(ticker), info)

    # -- manifest / housekeeping ------------------------------------------
    def manifest(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_path.is_file():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def delete(self, ticker: str) -> None:
        t = normalize_ticker(ticker)
        self.history_path(t).unlink(missing_ok=True)
        self.info_path(t).unlink(missing_ok=True)
        with self._lock:
            manifest = self.manifest()
            if manifest.pop(t, None) is not None:
                _atomic_write_json(self.manifest_path, manifest)

    def cached_tickers(self) -> list[str]:
        if not self.prices_dir.is_dir():
            return []
        return sorted(p.stem for p in self.prices_dir.glob("*.parquet"))

    def status(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """Download metadata per ticker; tickers not cached have empty fields."""
        manifest = self.manifest()
        names = [normalize_ticker(t) for t in tickers] if tickers is not None else self.cached_tickers()
        rows = []
        for t in names:
            entry = manifest.get(t, {}) if self.has_history(t) else {}
            rows.append({"ticker": t, "cached": bool(entry), **{c: entry.get(c) for c in STATUS_COLUMNS}})
        return pd.DataFrame(rows, columns=["ticker", "cached", *STATUS_COLUMNS]).set_index("ticker")

    def _update_manifest(self, ticker: str, entry: dict[str, Any]) -> None:
        with self._lock:
            manifest = self.manifest()
            manifest[ticker] = entry
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.manifest_path, dict(sorted(manifest.items())))


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
