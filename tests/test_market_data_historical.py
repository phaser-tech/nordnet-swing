"""Unit tests for historical ingestion. Never hits yfinance or the DB.

yfinance is monkeypatched; storage functions are monkeypatched where needed.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from packages.market_data import historical
from packages.market_data.historical import (
    DAILY_INTERVAL,
    Bar,
    _frame_to_bars,
    _normalize_yf_frame,
    _validate_interval,
    fetch_bars,
    sync,
)


def _yf_like_frame(*, multiindex: bool = False, ticker: str = "^OMXS30") -> pd.DataFrame:
    """Build a DataFrame shaped like yfinance's download output."""
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    data = {
        "Open": [100.0, 101.5, 102.0],
        "High": [101.0, 102.5, 103.0],
        "Low": [99.5, 100.5, 101.0],
        "Close": [100.5, 102.0, 102.5],
        "Adj Close": [100.5, 102.0, 102.5],
        "Volume": [1000, 1100, 1200],
    }
    frame = pd.DataFrame(data, index=index)
    if multiindex:
        # yfinance uses a (field, ticker) MultiIndex for single-ticker downloads.
        frame.columns = pd.MultiIndex.from_product([frame.columns, [ticker]])
    return frame


class TestValidateInterval:
    def test_accepts_daily(self) -> None:
        _validate_interval(DAILY_INTERVAL)  # no raise

    @pytest.mark.parametrize("interval", ["1m", "5m", "1h", "1wk"])
    def test_rejects_intraday_and_others(self, interval: str) -> None:
        with pytest.raises(ValueError, match="Unsupported interval"):
            _validate_interval(interval)


class TestBarModel:
    def test_is_frozen(self) -> None:
        bar = Bar(
            ticker="^OMXS30",
            interval=DAILY_INTERVAL,
            ts=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100.5"),
            volume=1000,
        )
        with pytest.raises(Exception):  # noqa: B017 - pydantic frozen raises ValidationError
            bar.close = Decimal("200")  # type: ignore[misc]

    def test_prices_are_decimal(self) -> None:
        bar = Bar(
            ticker="^NDX",
            interval=DAILY_INTERVAL,
            ts=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
            open=Decimal("100.25"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100.5"),
            volume=5,
        )
        assert isinstance(bar.open, Decimal)
        assert bar.open == Decimal("100.25")


class TestNormalizeYfFrame:
    def test_single_level_columns(self) -> None:
        out = _normalize_yf_frame(_yf_like_frame(), "^OMXS30")
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert "Adj Close" not in out.columns
        assert len(out) == 3

    def test_multiindex_columns_collapsed(self) -> None:
        out = _normalize_yf_frame(_yf_like_frame(multiindex=True), "^OMXS30")
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(out) == 3

    def test_index_localized_to_utc(self) -> None:
        out = _normalize_yf_frame(_yf_like_frame(), "^OMXS30")
        assert out.index.name == "ts"
        assert out.index.tz is not None
        assert str(out.index.tz) == "UTC"

    def test_drops_nan_rows(self) -> None:
        frame = _yf_like_frame()
        frame.iloc[1, frame.columns.get_loc("Close")] = np.nan
        out = _normalize_yf_frame(frame, "^OMXS30")
        assert len(out) == 2

    def test_empty_frame_returns_empty_canonical(self) -> None:
        out = _normalize_yf_frame(pd.DataFrame(), "^OMXS30")
        assert out.empty
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_missing_column_raises(self) -> None:
        frame = _yf_like_frame().drop(columns=["High"])
        with pytest.raises(ValueError, match="missing columns"):
            _normalize_yf_frame(frame, "^OMXS30")


class TestFrameToBars:
    def test_converts_to_decimal_and_int(self) -> None:
        frame = _normalize_yf_frame(_yf_like_frame(), "^OMXS30")
        bars = _frame_to_bars(frame, "^OMXS30", DAILY_INTERVAL)
        assert len(bars) == 3
        first = bars[0]
        assert isinstance(first.open, Decimal)
        assert first.open == Decimal("100.0")
        assert first.volume == 1000
        assert first.ticker == "^OMXS30"
        assert first.interval == DAILY_INTERVAL

    def test_decimal_has_no_float_artifacts(self) -> None:
        frame = _normalize_yf_frame(_yf_like_frame(), "^OMXS30")
        bars = _frame_to_bars(frame, "^OMXS30", DAILY_INTERVAL)
        # 101.5 must not become 101.5000000000000284...
        assert bars[1].open == Decimal("101.5")


class TestFetchBars:
    def test_calls_yfinance_and_normalizes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_download(**kwargs: object) -> pd.DataFrame:
            captured.update(kwargs)
            return _yf_like_frame(multiindex=True)

        monkeypatch.setattr(historical.yf, "download", fake_download)
        out = fetch_bars("^OMXS30", date(2024, 1, 1), date(2024, 2, 1))

        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(out) == 3
        # auto_adjust disabled so we store raw OHLC.
        assert captured["auto_adjust"] is False
        assert captured["interval"] == DAILY_INTERVAL

    def test_empty_download_returns_empty_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(historical.yf, "download", lambda **_: pd.DataFrame())
        out = fetch_bars("^OMXS30", date(2024, 1, 1), date(2024, 2, 1))
        assert out.empty

    def test_rejects_intraday_before_calling_yfinance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(**_: object) -> pd.DataFrame:
            raise AssertionError("yfinance must not be called for bad interval")

        monkeypatch.setattr(historical.yf, "download", boom)
        with pytest.raises(ValueError, match="Unsupported interval"):
            fetch_bars("^OMXS30", date(2024, 1, 1), date(2024, 2, 1), interval="1m")


class TestSync:
    def test_iterates_tickers_and_aggregates_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetch_calls: list[str] = []
        store_calls: list[str] = []

        def fake_fetch(
            ticker: str, start: object, end: object, interval: str
        ) -> pd.DataFrame:
            fetch_calls.append(ticker)
            return _normalize_yf_frame(_yf_like_frame(), "^OMXS30")

        def fake_store(bars: pd.DataFrame, ticker: str, interval: str) -> int:
            store_calls.append(ticker)
            return len(bars)

        monkeypatch.setattr(historical, "fetch_bars", fake_fetch)
        monkeypatch.setattr(historical, "store_bars", fake_store)

        results = sync(("^OMXS30", "^NDX"), date(2024, 1, 1), end_date=date(2024, 2, 1))

        assert fetch_calls == ["^OMXS30", "^NDX"]
        assert store_calls == ["^OMXS30", "^NDX"]
        assert results == {"^OMXS30": 3, "^NDX": 3}
