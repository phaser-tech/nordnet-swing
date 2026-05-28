"""Unit tests for the wide-frame assembly helper. Injected loader; no DB."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from packages.backtest.assembly import assemble_cross_asset_frame


def _frame(dates: list[str], close: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="ts").tz_localize("UTC")
    return pd.DataFrame(
        {
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": [1000] * len(close),
        },
        index=idx,
    )


def _loader(frames: dict[str, pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    def load(ticker: str, start: object, end: object, interval: str) -> pd.DataFrame:
        return frames[ticker]

    return load


def test_assembles_all_cross_asset_columns() -> None:
    omx_dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    frames = {
        "^OMX": _frame(omx_dates, [100, 101, 102, 103]),
        "^GSPC": _frame(omx_dates, [4000, 4010, 4020, 4030]),
        "^VIX": _frame(omx_dates, [20, 19, 18, 17]),
        "^TNX": _frame(omx_dates, [40, 41, 42, 43]),
        "DX-Y.NYB": _frame(omx_dates, [104, 103, 102, 101]),
        "GC=F": _frame(omx_dates, [2000, 2010, 2020, 2030]),
    }
    wide = assemble_cross_asset_frame(
        loader=_loader(frames),
        instrument="^OMX",
        start="2024-01-01",
        end="2024-01-06",
        interval="1d",
    )
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in wide.columns
    for col in ("SPX_Close", "VIX_Close", "TNX_Close", "DXY_Close", "GOLD_Close"):
        assert col in wide.columns
    assert len(wide) == 4
    assert wide["SPX_Close"].tolist() == [4000, 4010, 4020, 4030]


def test_foreign_holiday_is_forward_filled() -> None:
    omx_dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    # SPX missing 01-04 (US holiday); most recent close (01-03) must carry forward.
    spx = _frame(["2024-01-02", "2024-01-03", "2024-01-05"], [4000, 4010, 4030])
    frames = {
        "^OMX": _frame(omx_dates, [100, 101, 102, 103]),
        "^GSPC": spx,
        "^VIX": _frame(omx_dates, [20, 19, 18, 17]),
        "^TNX": _frame(omx_dates, [40, 41, 42, 43]),
        "DX-Y.NYB": _frame(omx_dates, [104, 103, 102, 101]),
        "GC=F": _frame(omx_dates, [2000, 2010, 2020, 2030]),
    }
    wide = assemble_cross_asset_frame(
        loader=_loader(frames),
        instrument="^OMX",
        start="2024-01-01",
        end="2024-01-06",
        interval="1d",
    )
    assert len(wide) == 4  # aligned to OMX calendar
    holiday = pd.Timestamp("2024-01-04", tz="UTC")
    assert wide.loc[holiday, "SPX_Close"] == 4010  # carried from 01-03


def test_empty_instrument_returns_empty() -> None:
    frames = {"^OMX": _frame([], []).iloc[0:0]}
    wide = assemble_cross_asset_frame(
        loader=_loader(frames),
        instrument="^OMX",
        start="2024-01-01",
        end="2024-01-06",
        interval="1d",
    )
    assert wide.empty
