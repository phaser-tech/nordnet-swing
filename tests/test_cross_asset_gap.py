"""Unit tests for the cross-asset gap-capture strategy. Fixtures only."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from packages.core.domain.signal import Conviction, Direction, Signal
from packages.strategies.cross_asset_gap import (
    GAP_DIRECTION_COL,
    CrossAssetGapStrategy,
    build_per_day_gap,
)

_DATES = [
    "2024-01-02",
    "2024-01-03",
    "2024-01-04",
    "2024-01-05",
]
_OPENS = [100.0, 105.0, 107.0, 103.0]
_CLOSES = [101.0, 106.0, 108.0, 104.0]


def _wide_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(_DATES), name="ts").tz_localize("UTC")
    return pd.DataFrame(
        {
            "Open": _OPENS,
            "High": [c + 1 for c in _CLOSES],
            "Low": [o - 1 for o in _OPENS],
            "Close": _CLOSES,
            "Volume": [1000] * len(_DATES),
        },
        index=idx,
    )


def _signal(ts: pd.Timestamp, direction: Direction) -> Signal:
    return Signal(
        timestamp=ts.to_pydatetime(),
        strategy_name="cross_asset_confluence",  # the source the script uses
        instrument="^OMX",
        direction=direction,
        conviction=Conviction.HIGH,
        suggested_entry=Decimal("100"),
        suggested_stop=Decimal("99"),
        suggested_target=Decimal("101"),
    )


class TestBuildPerDayGap:
    def test_empty_frame_returns_empty_synthetic(self) -> None:
        out = build_per_day_gap(pd.DataFrame(), [])
        assert out.empty
        assert GAP_DIRECTION_COL in out.columns

    def test_single_row_frame_is_too_short_for_a_gap(self) -> None:
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC")], name="ts")
        one = pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0], "Volume": [0]},
            index=idx,
        )
        out = build_per_day_gap(one, [])
        assert out.empty

    def test_first_day_is_dropped_so_no_lookback_undefined(self) -> None:
        out = build_per_day_gap(_wide_frame(), [])
        assert pd.Timestamp(_DATES[0], tz="UTC") not in out.index
        assert len(out) == len(_DATES) - 1

    def test_signal_day_records_gap_legs_and_direction(self) -> None:
        # Signal fires on 2024-01-03 LONG. Open should be 2024-01-02's Close
        # (101.0), Close should be 2024-01-03's Open (105.0) -- so the signed
        # gap return is (105 - 101) / 101 = +3.96%.
        signal_ts = pd.Timestamp("2024-01-03", tz="UTC")
        out = build_per_day_gap(_wide_frame(), [_signal(signal_ts, Direction.LONG)])
        row = out.loc[signal_ts]
        assert row[GAP_DIRECTION_COL] is Direction.LONG
        assert row["Open"] == pytest.approx(101.0)
        assert row["Close"] == pytest.approx(105.0)

    def test_short_signal_keeps_raw_legs_unchanged(self) -> None:
        # SHORT means we *sell* at close(T-1) and *cover* at open(T). The
        # synthetic row keeps the raw legs (Open=close_prev, Close=open_t) --
        # oos.py applies the sign via the signal direction itself.
        signal_ts = pd.Timestamp("2024-01-04", tz="UTC")
        out = build_per_day_gap(_wide_frame(), [_signal(signal_ts, Direction.SHORT)])
        row = out.loc[signal_ts]
        assert row[GAP_DIRECTION_COL] is Direction.SHORT
        assert row["Open"] == pytest.approx(106.0)  # close on 2024-01-03
        assert row["Close"] == pytest.approx(107.0)  # open on 2024-01-04

    def test_non_signal_day_has_placeholder_open_equals_close(self) -> None:
        # No signal on 2024-01-04 -> Open == Close == close_prev (no synthetic
        # move, so even if oos.py accidentally read it the contribution is 0).
        out = build_per_day_gap(_wide_frame(), [])
        row = out.loc[pd.Timestamp("2024-01-04", tz="UTC")]
        assert row[GAP_DIRECTION_COL] is None
        assert row["Open"] == row["Close"] == pytest.approx(106.0)

    def test_signal_on_dropped_first_day_is_ignored(self) -> None:
        # Signal on 2024-01-02 but that day is the dropped first day -> no row.
        first_day = pd.Timestamp(_DATES[0], tz="UTC")
        out = build_per_day_gap(_wide_frame(), [_signal(first_day, Direction.LONG)])
        assert first_day not in out.index
        # The remaining rows are all no-trade rows.
        for ts in out.index:
            assert out.loc[ts, GAP_DIRECTION_COL] is None

    def test_multiple_signals_each_get_their_own_row(self) -> None:
        out = build_per_day_gap(
            _wide_frame(),
            [
                _signal(pd.Timestamp("2024-01-03", tz="UTC"), Direction.LONG),
                _signal(pd.Timestamp("2024-01-04", tz="UTC"), Direction.SHORT),
            ],
        )
        assert out.loc[pd.Timestamp("2024-01-03", tz="UTC"), GAP_DIRECTION_COL] is Direction.LONG
        assert out.loc[pd.Timestamp("2024-01-04", tz="UTC"), GAP_DIRECTION_COL] is Direction.SHORT


class TestStrategyEmitter:
    def test_emits_one_signal_per_set_direction(self) -> None:
        synthetic = build_per_day_gap(
            _wide_frame(),
            [
                _signal(pd.Timestamp("2024-01-03", tz="UTC"), Direction.LONG),
                _signal(pd.Timestamp("2024-01-04", tz="UTC"), Direction.SHORT),
            ],
        )
        sigs = list(CrossAssetGapStrategy("^OMX").generate_signals(synthetic))
        assert len(sigs) == 2
        ts_to_dir = {s.timestamp: s.direction for s in sigs}
        assert ts_to_dir[datetime.fromisoformat("2024-01-03T00:00:00+00:00")] == Direction.LONG
        assert ts_to_dir[datetime.fromisoformat("2024-01-04T00:00:00+00:00")] == Direction.SHORT
        assert sigs[0].instrument == "^OMX"

    def test_skips_no_trade_rows(self) -> None:
        synthetic = build_per_day_gap(_wide_frame(), [])
        sigs = list(CrossAssetGapStrategy("^OMX").generate_signals(synthetic))
        assert sigs == []

    def test_rejects_frame_without_gap_direction_column(self) -> None:
        plain = _wide_frame()
        with pytest.raises(ValueError, match=GAP_DIRECTION_COL):
            list(CrossAssetGapStrategy("^OMX").generate_signals(plain))

    def test_entry_price_matches_synthetic_open(self) -> None:
        signal_ts = pd.Timestamp("2024-01-03", tz="UTC")
        synthetic = build_per_day_gap(_wide_frame(), [_signal(signal_ts, Direction.LONG)])
        sigs = list(CrossAssetGapStrategy("^OMX").generate_signals(synthetic))
        # On the signal day, synthetic Open is close(T-1) = 101.0. The emitted
        # Signal carries that as suggested_entry (advisory; oos.py reads the
        # synthetic Open directly).
        assert float(sigs[0].suggested_entry) == pytest.approx(101.0)
