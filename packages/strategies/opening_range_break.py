"""Opening-range break (ORB) on intraday bars (issue #19).

Define the "opening range" as the high/low of the 09:00 Stockholm bar. We wait
for a subsequent bar to *close* outside that range, then trade the break
direction from the next bar's open through the close of the last full bar of
the session. The 17:00 Stockholm bar is a 30-min partial covering the
17:00-17:30 close and is intentionally not held -- this keeps the trade inside
the no-overnight rule with margin (and skips the only fill where the partial
bar muddies the price).

Architectural seam (pre-registered in #19): the strategy's natural input is 1h
bars, but the OOS harness scores `sign * (close - open) / open` per signal bar.
We collapse the 1h bars into a synthetic *per-day* frame where each row's
Open/Close are the actual entry/exit prices for that day's ORB trade (and
placeholders on no-trade days). `oos.py` is left untouched -- it sees one bar
per day, exactly as for any daily strategy.

LOOKAHEAD SAFETY: the decision (direction + entry bar) is locked at the close
of the confirming bar. Entry is the open of the *next* bar. The exit bar's
close is the realised exit, not consulted by the entry decision. Tests mutate
post-entry bars and assert direction/entry unchanged.

Stateless: instrument + min_trailing_bars are configuration. Imports only
`core` + pandas (dependency rule #6).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date as date_type
from decimal import Decimal
from typing import cast

import pandas as pd

from packages.core.domain.signal import Conviction, Direction, Signal

STOCKHOLM_TZ = "Europe/Stockholm"
RANGE_BAR_HOUR = 9  # 09:00 Stockholm -- the first full bar of the session
PARTIAL_BAR_HOUR = 17  # 17:00 Stockholm -- the 30-min partial close bar; never held
ORB_DIRECTION_COL = "_orb_direction"
# Diagnostic columns: not read by oos.py, used by the run script to decompose
# break-direction frequency, hold length, time-of-day, and the "is the range
# really just the overnight gap?" first-bar return distribution.
ORB_CONFIRM_HOUR_COL = "_orb_confirm_se_hour"
ORB_BARS_HELD_COL = "_orb_bars_held"
ORB_FIRST_BAR_RETURN_COL = "_orb_first_bar_return"
_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_OUTPUT_COLUMNS = (
    *_OHLCV_COLUMNS,
    ORB_DIRECTION_COL,
    ORB_CONFIRM_HOUR_COL,
    ORB_BARS_HELD_COL,
    ORB_FIRST_BAR_RETURN_COL,
)


def build_per_day_orb(
    bars: pd.DataFrame,
    *,
    min_trailing_bars: int = 3,
) -> pd.DataFrame:
    """Collapse 1h bars into the synthetic per-day frame `oos.py` evaluates.

    Per Stockholm trading day, find the 09:00 range bar, then look for the
    first subsequent bar (excluding the 17:00 partial) whose close exits the
    range. If found AND there is a further bar to enter on, the row records:

        Open  = entry-bar open   (the bar AFTER the confirming close)
        Close = exit-bar close   (the last full bar of the day)
        _orb_direction = Direction.LONG | Direction.SHORT

    For days without a confirmed-and-actionable break, the row records
    placeholder Open/Close (the last full bar's, never read by `oos.py` since
    there is no signal there) and `_orb_direction = None`.

    Days with fewer than `min_trailing_bars` after the range bar (after the
    17:00 partial is dropped) are skipped entirely, removing half-day fragments
    from the equity curve.

    The frame is indexed by UTC midnight of the trading date so the strategy's
    signal timestamps line up with the synthetic-frame index exactly.
    """
    if bars.empty:
        return _empty_per_day_frame()
    local = bars.tz_convert(STOCKHOLM_TZ)
    rows: list[dict[str, object]] = []
    for day_key, day_bars in local.groupby(local.index.date):
        day_date = _as_date(day_key)
        row = _build_day_row(day_date, day_bars, min_trailing_bars=min_trailing_bars)
        if row is not None:
            rows.append(row)
    if not rows:
        return _empty_per_day_frame()
    frame = pd.DataFrame(rows).set_index("ts")
    frame.index = pd.DatetimeIndex(frame.index, tz="UTC", name="ts")
    return frame[list(_OUTPUT_COLUMNS)]


def _empty_per_day_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(_OUTPUT_COLUMNS))
    frame.index = pd.DatetimeIndex([], tz="UTC", name="ts")
    return frame


def _as_date(value: object) -> date_type:
    """Groupby keys from `index.date` are usually `datetime.date`; coerce
    defensively in case pandas hands back a numpy date64 in some versions."""
    if isinstance(value, date_type):
        return value
    return cast(date_type, pd.Timestamp(value).date())


def _build_day_row(
    day_date: date_type,
    day_bars: pd.DataFrame,
    *,
    min_trailing_bars: int,
) -> dict[str, object] | None:
    range_bar = day_bars[day_bars.index.hour == RANGE_BAR_HOUR]
    if range_bar.empty:
        return None  # no 09:00 bar; incomplete day

    range_ts = range_bar.index[0]
    range_high = float(range_bar["High"].iloc[0])
    range_low = float(range_bar["Low"].iloc[0])

    trailing = day_bars[
        (day_bars.index > range_ts) & (day_bars.index.hour != PARTIAL_BAR_HOUR)
    ]
    if len(trailing) < min_trailing_bars:
        return None

    ts_utc = pd.Timestamp(day_date, tz="UTC")
    last_full = trailing.iloc[-1]
    range_open = float(range_bar["Open"].iloc[0])
    range_close = float(range_bar["Close"].iloc[0])
    first_bar_return = (range_close - range_open) / range_open if range_open != 0.0 else 0.0

    direction: Direction | None = None
    confirm_position = -1
    for i in range(len(trailing)):
        close_i = float(trailing["Close"].iloc[i])
        if close_i > range_high:
            direction = Direction.LONG
            confirm_position = i
            break
        if close_i < range_low:
            direction = Direction.SHORT
            confirm_position = i
            break

    # If the break confirms at the very last full bar of the day there is no
    # bar to enter on, so demote to no-trade for this day.
    if direction is not None and confirm_position + 1 >= len(trailing):
        direction = None

    if direction is None:
        return {
            "ts": ts_utc,
            "Open": float(last_full["Open"]),
            "High": float(last_full["High"]),
            "Low": float(last_full["Low"]),
            "Close": float(last_full["Close"]),
            "Volume": 0,
            ORB_DIRECTION_COL: None,
            ORB_CONFIRM_HOUR_COL: None,
            ORB_BARS_HELD_COL: 0,
            ORB_FIRST_BAR_RETURN_COL: first_bar_return,
        }

    entry_position = confirm_position + 1
    entry_open = float(trailing["Open"].iloc[entry_position])
    exit_close = float(last_full["Close"])
    confirm_se_hour = int(trailing.index[confirm_position].hour)
    bars_held = len(trailing) - entry_position  # entry .. last_full inclusive
    return {
        "ts": ts_utc,
        "Open": entry_open,
        "High": max(entry_open, exit_close),  # placeholder; unused by oos.py
        "Low": min(entry_open, exit_close),
        "Close": exit_close,
        "Volume": 0,
        ORB_DIRECTION_COL: direction,
        ORB_CONFIRM_HOUR_COL: confirm_se_hour,
        ORB_BARS_HELD_COL: bars_held,
        ORB_FIRST_BAR_RETURN_COL: first_bar_return,
    }


class OpeningRangeBreakStrategy:
    """Thin emitter over the synthetic per-day frame -- one Signal per break day."""

    def __init__(self, instrument: str) -> None:
        self._instrument = instrument

    @property
    def name(self) -> str:
        return "opening_range_break"

    def generate_signals(self, market_data: pd.DataFrame) -> Iterator[Signal]:
        if ORB_DIRECTION_COL not in market_data.columns:
            raise ValueError(
                f"market_data must be the synthetic per-day frame produced by "
                f"build_per_day_orb (missing column {ORB_DIRECTION_COL!r})."
            )
        for ts, row in market_data.iterrows():
            direction = row[ORB_DIRECTION_COL]
            if not isinstance(direction, Direction):
                continue  # None (no-trade day) or accidental NaN coercion
            entry = Decimal(str(row["Open"]))
            target_mult = Decimal("1.01") if direction == Direction.LONG else Decimal("0.99")
            stop_mult = Decimal("0.99") if direction == Direction.LONG else Decimal("1.01")
            yield Signal(
                timestamp=pd.Timestamp(ts).to_pydatetime(),
                strategy_name=self.name,
                instrument=self._instrument,
                direction=direction,
                conviction=Conviction.HIGH,
                suggested_entry=entry,
                suggested_stop=entry * stop_mult,
                suggested_target=entry * target_mult,
                confluence_factors=["orb_break"],
                notes=f"ORB {direction.value} break (open->close)",
            )
