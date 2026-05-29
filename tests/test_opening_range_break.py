"""Unit tests for the opening-range break strategy. Fixtures only."""

from __future__ import annotations

import pandas as pd
import pytest

from packages.core.domain.signal import Direction, Signal
from packages.strategies.opening_range_break import (
    ORB_BARS_HELD_COL,
    ORB_CONFIRM_HOUR_COL,
    ORB_DIRECTION_COL,
    ORB_FIRST_BAR_RETURN_COL,
    OpeningRangeBreakStrategy,
    build_per_day_orb,
)

# A Tuesday in June -- CEST means UTC 07:00 == Stockholm 09:00, so a 9-bar
# day runs UTC 07:00..15:00 (Stockholm 09:00..17:00).
_SUMMER_DATE = "2025-06-03"
_WINTER_DATE = "2025-12-02"
# Per the in-range vs break-confirmed scenarios below, the range bar has
# High=101, Low=99 so closes outside [99, 101] confirm a break.
_RANGE_BAR_OHLC = (100.0, 101.0, 99.0, 100.5)


def _bars(
    *,
    closes: list[float],
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    date_str: str = _SUMMER_DATE,
    utc_start_hour: int = 7,
) -> pd.DataFrame:
    """Build a single-day OHLCV frame: 9 hourly bars at UTC `utc_start_hour`..+8.

    With `_SUMMER_DATE` and start hour 7 the bars line up exactly to Stockholm
    09:00..17:00 local. With `_WINTER_DATE` and start hour 8, ditto.
    """
    n = len(closes)
    opens = opens if opens is not None else closes
    highs = highs if highs is not None else [max(o, c) for o, c in zip(opens, closes, strict=True)]
    lows = lows if lows is not None else [min(o, c) for o, c in zip(opens, closes, strict=True)]
    index = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{date_str} {utc_start_hour + i:02d}:00:00", tz="UTC")
            for i in range(n)
        ],
        name="ts",
    )
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [0] * n},
        index=index,
    )


def _with_range_bar(
    *,
    trailing_closes: list[float],
    trailing_opens: list[float] | None = None,
    trailing_highs: list[float] | None = None,
    trailing_lows: list[float] | None = None,
    partial_close: float | None = 100.0,
    date_str: str = _SUMMER_DATE,
    utc_start_hour: int = 7,
) -> pd.DataFrame:
    """Build a full day: 1 range bar + N trailing full bars + optional 17:00 partial."""
    ro, rh, rl, rc = _RANGE_BAR_OHLC
    opens = [ro, *(trailing_opens or trailing_closes)]
    highs = [rh, *(trailing_highs or [max(o, c) for o, c in zip(opens[1:], trailing_closes, strict=True)])]
    lows = [rl, *(trailing_lows or [min(o, c) for o, c in zip(opens[1:], trailing_closes, strict=True)])]
    closes = [rc, *trailing_closes]
    if partial_close is not None:
        opens.append(partial_close)
        highs.append(partial_close)
        lows.append(partial_close)
        closes.append(partial_close)
    return _bars(
        closes=closes,
        opens=opens,
        highs=highs,
        lows=lows,
        date_str=date_str,
        utc_start_hour=utc_start_hour,
    )


def _row(frame: pd.DataFrame, date_str: str) -> pd.Series:
    return frame.loc[pd.Timestamp(date_str, tz="UTC")]


class TestBuildPerDayOrbBasics:
    def test_empty_input_returns_empty_frame(self) -> None:
        empty = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([], tz="UTC", name="ts"),
        )
        out = build_per_day_orb(empty)
        assert out.empty
        assert ORB_DIRECTION_COL in out.columns

    def test_synthetic_index_is_utc_midnight_of_local_date(self) -> None:
        # Range [99, 101]; one bar at SE 10 breaks LONG at C=102; bar at SE 11
        # is the entry. The synthetic row is timestamped 2025-06-03 00:00 UTC.
        bars = _with_range_bar(
            trailing_closes=[102.0, 100.5, 100.5, 100.5, 100.5, 100.5, 105.0],
        )
        out = build_per_day_orb(bars)
        assert list(out.index) == [pd.Timestamp(_SUMMER_DATE, tz="UTC")]


class TestBreakDetection:
    def test_upward_break_emits_long(self) -> None:
        bars = _with_range_bar(
            trailing_opens=[101.5, 102.5, 103.0, 103.2, 104.0, 104.5, 105.0],
            trailing_closes=[102.0, 103.0, 103.2, 104.0, 104.5, 105.0, 105.0],
        )
        out = build_per_day_orb(bars)
        row = _row(out, _SUMMER_DATE)
        assert row[ORB_DIRECTION_COL] is Direction.LONG
        # Entry = open of the bar AFTER the confirming close (SE 11 open).
        assert row["Open"] == pytest.approx(102.5)
        # Exit = close of the last full bar (SE 16).
        assert row["Close"] == pytest.approx(105.0)

    def test_downward_break_emits_short(self) -> None:
        bars = _with_range_bar(
            trailing_opens=[100.0, 97.5, 97.0, 97.0, 96.5, 96.0, 95.5],
            trailing_closes=[98.0, 97.5, 97.0, 96.5, 96.0, 95.5, 95.0],
        )
        out = build_per_day_orb(bars)
        row = _row(out, _SUMMER_DATE)
        assert row[ORB_DIRECTION_COL] is Direction.SHORT
        assert row["Open"] == pytest.approx(97.5)  # SE 11 open
        assert row["Close"] == pytest.approx(95.0)  # SE 16 close

    def test_no_break_marks_day_as_no_trade(self) -> None:
        # All trailing closes stay within [99, 101].
        bars = _with_range_bar(
            trailing_opens=[100.0] * 7,
            trailing_closes=[100.2, 100.4, 100.1, 99.8, 100.3, 99.9, 100.5],
        )
        out = build_per_day_orb(bars)
        row = _row(out, _SUMMER_DATE)
        assert row[ORB_DIRECTION_COL] is None

    def test_only_first_break_wins(self) -> None:
        # SE 10 closes 102 (LONG break). SE 13 closes 95 (would be SHORT). The
        # SHORT must NOT override the locked-in LONG decision.
        bars = _with_range_bar(
            trailing_opens=[101.0, 102.5, 103.0, 96.0, 95.0, 95.0, 95.0],
            trailing_closes=[102.0, 103.0, 95.0, 95.5, 95.5, 95.5, 95.0],
        )
        out = build_per_day_orb(bars)
        row = _row(out, _SUMMER_DATE)
        assert row[ORB_DIRECTION_COL] is Direction.LONG
        assert row["Open"] == pytest.approx(102.5)  # SE 11 open

    def test_break_at_last_full_bar_demotes_to_no_trade(self) -> None:
        # Closes stay in range for SE 10..15; SE 16 breaks LONG. No bar after
        # the 16:00 confirm to enter on (the 17:00 bar is the dropped partial).
        bars = _with_range_bar(
            trailing_opens=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            trailing_closes=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 102.0],
        )
        out = build_per_day_orb(bars)
        row = _row(out, _SUMMER_DATE)
        assert row[ORB_DIRECTION_COL] is None


class TestDiagnosticColumns:
    def test_break_day_records_confirm_hour_bars_held_first_bar_return(self) -> None:
        # Range bar Open=100, Close=100.5 -> first-bar return = +0.5%.
        # Confirming close at SE 10 (index 1 overall, position 0 in trailing).
        # Entry at SE 11 (trailing position 1). 7 trailing bars (SE 10..16);
        # bars_held = 7 - 1 = 6 (SE 11..16 inclusive).
        bars = _with_range_bar(
            trailing_opens=[101.0, 102.5, 103.0, 103.0, 103.0, 103.0, 103.0],
            trailing_closes=[102.0, 103.0, 103.0, 103.0, 103.0, 103.0, 104.0],
        )
        row = _row(build_per_day_orb(bars), _SUMMER_DATE)
        assert row[ORB_CONFIRM_HOUR_COL] == 10
        assert row[ORB_BARS_HELD_COL] == 6
        assert row[ORB_FIRST_BAR_RETURN_COL] == pytest.approx((100.5 - 100.0) / 100.0)

    def test_no_trade_day_records_first_bar_return_with_no_confirm(self) -> None:
        bars = _with_range_bar(
            trailing_opens=[100.0] * 7,
            trailing_closes=[100.0] * 7,
        )
        row = _row(build_per_day_orb(bars), _SUMMER_DATE)
        assert row[ORB_CONFIRM_HOUR_COL] is None
        assert row[ORB_BARS_HELD_COL] == 0
        assert row[ORB_FIRST_BAR_RETURN_COL] == pytest.approx((100.5 - 100.0) / 100.0)


class TestPartialAndShortDays:
    def test_17_00_partial_is_excluded_from_break_search(self) -> None:
        # All full-bar closes stay in range; the 17:00 partial's close jumps to
        # 105. If we accidentally consulted it we'd record a LONG.
        bars = _with_range_bar(
            trailing_closes=[100.0] * 7,
            partial_close=105.0,
        )
        out = build_per_day_orb(bars)
        row = _row(out, _SUMMER_DATE)
        assert row[ORB_DIRECTION_COL] is None

    def test_short_day_below_min_trailing_is_skipped(self) -> None:
        # Range bar + 2 trailing full bars (no partial) -> 2 < min_trailing 3.
        bars = _with_range_bar(
            trailing_opens=[101.0, 102.0],
            trailing_closes=[102.0, 102.0],  # would be LONG if it were eligible
            partial_close=None,
        )
        out = build_per_day_orb(bars)
        assert out.empty

    def test_short_day_at_min_trailing_is_kept(self) -> None:
        bars = _with_range_bar(
            trailing_opens=[101.0, 102.0, 102.0],
            trailing_closes=[102.0, 102.0, 102.0],
            partial_close=None,
        )
        out = build_per_day_orb(bars, min_trailing_bars=3)
        assert len(out) == 1
        assert _row(out, _SUMMER_DATE)[ORB_DIRECTION_COL] is Direction.LONG


class TestDST:
    def test_handles_winter_bars_correctly(self) -> None:
        # In CET (UTC+1) the 09:00 Stockholm bar is at 08:00 UTC.
        bars = _with_range_bar(
            trailing_opens=[101.0, 102.5, 103.0, 103.0, 103.0, 103.0, 103.0],
            trailing_closes=[102.0, 103.0, 103.0, 103.0, 103.0, 103.0, 104.0],
            date_str=_WINTER_DATE,
            utc_start_hour=8,
        )
        out = build_per_day_orb(bars)
        assert list(out.index) == [pd.Timestamp(_WINTER_DATE, tz="UTC")]
        row = _row(out, _WINTER_DATE)
        assert row[ORB_DIRECTION_COL] is Direction.LONG
        assert row["Open"] == pytest.approx(102.5)
        assert row["Close"] == pytest.approx(104.0)


class TestNoLookahead:
    def test_post_entry_bars_do_not_change_decision(self) -> None:
        # LONG break confirms at SE 10 (index 1); entry at SE 11 (index 2);
        # exit at SE 16 (index 7). Bars at indices 3..6 are between entry and
        # exit -- mutating them must not change direction, entry, or exit.
        base = _with_range_bar(
            trailing_opens=[101.0, 102.5, 103.0, 103.2, 104.0, 104.5, 105.0],
            trailing_closes=[102.0, 103.0, 103.2, 104.0, 104.5, 105.0, 105.0],
        )
        baseline = _row(build_per_day_orb(base), _SUMMER_DATE)

        # Mutate the SE 13 bar (full index 4) Close to a wild value.
        mutated = base.copy()
        ts_se13 = base.index[4]
        mutated.loc[ts_se13, "Close"] = 500.0
        after = _row(build_per_day_orb(mutated), _SUMMER_DATE)

        assert after[ORB_DIRECTION_COL] is baseline[ORB_DIRECTION_COL]
        assert after["Open"] == pytest.approx(baseline["Open"])
        assert after["Close"] == pytest.approx(baseline["Close"])


class TestStrategyEmitter:
    def test_emits_one_signal_per_break_row(self) -> None:
        bars = _with_range_bar(
            trailing_opens=[101.0, 102.5, 103.0, 103.0, 103.0, 103.0, 103.0],
            trailing_closes=[102.0, 103.0, 103.0, 103.0, 103.0, 103.0, 104.0],
        )
        synthetic = build_per_day_orb(bars)
        sigs = list(OpeningRangeBreakStrategy("^OMX").generate_signals(synthetic))
        assert len(sigs) == 1
        sig: Signal = sigs[0]
        assert sig.direction == Direction.LONG
        assert sig.instrument == "^OMX"
        assert sig.timestamp == pd.Timestamp(_SUMMER_DATE, tz="UTC").to_pydatetime()
        assert float(sig.suggested_entry) == pytest.approx(102.5)

    def test_skips_no_trade_rows(self) -> None:
        bars = _with_range_bar(
            trailing_opens=[100.0] * 7,
            trailing_closes=[100.0] * 7,
        )
        synthetic = build_per_day_orb(bars)
        assert len(synthetic) == 1
        sigs = list(OpeningRangeBreakStrategy("^OMX").generate_signals(synthetic))
        assert sigs == []

    def test_rejects_frame_without_direction_column(self) -> None:
        plain = pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0], "Volume": [0]},
            index=pd.DatetimeIndex([pd.Timestamp(_SUMMER_DATE, tz="UTC")], name="ts"),
        )
        with pytest.raises(ValueError, match=ORB_DIRECTION_COL):
            list(OpeningRangeBreakStrategy("^OMX").generate_signals(plain))
