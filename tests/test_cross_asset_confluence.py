"""Unit tests for the cross-asset confluence strategy.

Covers signal scoring (deadband), the confluence threshold, end-to-end
LONG/SHORT/flat behavior, and — critically — the no-lookahead guarantee.
Fixture data only; no DB/network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from packages.core.domain.signal import Conviction, Direction
from packages.strategies.cross_asset_confluence import (
    CrossAssetConfluenceStrategy,
    _confluence,
    _threshold,
)

# Tiny windows so fixtures stay small. warmup = max(sma, zwindow + change_lb) = 7.
_PARAMS = dict(
    spx_sma_window=3,
    zscore_window=5,
    deadband_k=0.3,
    min_agree=4,
    change_lookback=2,
    expected_move_window=3,
)


def _wide(
    spx: list[float],
    vix: list[float],
    tnx: list[float],
    dxy: list[float],
    gold: list[float],
    omx_close: list[float] | None = None,
) -> pd.DataFrame:
    n = len(spx)
    idx = pd.DatetimeIndex(
        pd.to_datetime([f"2024-01-{i + 1:02d}" for i in range(n)]), name="ts"
    ).tz_localize("UTC")
    close = omx_close if omx_close is not None else [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [c - 0.5 for c in close],
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": [1000] * n,
            "SPX_Close": spx,
            "VIX_Close": vix,
            "TNX_Close": tnx,
            "DXY_Close": dxy,
            "GOLD_Close": gold,
        },
        index=idx,
    )


# A 12-row frame: quiet for 9 days, then a decisive RISK-ON shock (SPX up, VIX
# collapsing, yields rising, USD + gold falling). Signals only fire once the
# shock lands -> exercises selectivity too.
_FLAT9 = [20.0] * 9


def _risk_on_frame() -> pd.DataFrame:
    return _wide(
        spx=[100.0 + i for i in range(12)],  # steady uptrend -> above SMA (+1)
        vix=[*_FLAT9, 16.0, 11.0, 7.0],  # collapsing -> risk-on
        tnx=[*([100.0] * 9), 101.0, 103.0, 106.0],  # rising yields -> risk-on
        dxy=[*([100.0] * 9), 99.0, 96.0, 93.0],  # falling USD -> risk-on
        gold=[*([100.0] * 9), 99.0, 96.0, 93.0],  # falling gold -> risk-on
    )


def _risk_off_frame() -> pd.DataFrame:
    return _wide(
        spx=[111.0 - i for i in range(12)],  # steady downtrend -> below SMA (-1)
        vix=[*_FLAT9, 25.0, 31.0, 38.0],  # spiking -> risk-off
        tnx=[*([100.0] * 9), 99.0, 97.0, 94.0],  # falling yields -> risk-off
        dxy=[*([100.0] * 9), 101.0, 104.0, 107.0],  # surging USD -> risk-off
        gold=[*([100.0] * 9), 101.0, 104.0, 107.0],  # surging gold -> risk-off
        omx_close=[111.0 - i for i in range(12)],
    )


class TestThreshold:
    def test_deadband(self) -> None:
        assert _threshold(1.0, 0.5) == 1
        assert _threshold(-1.0, 0.5) == -1
        assert _threshold(0.2, 0.5) == 0
        assert _threshold(0.5, 0.5) == 0  # strict inequality
        assert _threshold(float("nan"), 0.5) == 0


class TestConfluence:
    def test_long_at_threshold(self) -> None:
        assert _confluence([1, 1, 1, 1, 0], 4) == (Direction.LONG, 4)

    def test_long_unanimous(self) -> None:
        assert _confluence([1, 1, 1, 1, 1], 4) == (Direction.LONG, 5)

    def test_short_at_threshold(self) -> None:
        assert _confluence([-1, -1, -1, -1, 0], 4) == (Direction.SHORT, 4)

    @pytest.mark.parametrize(
        "scores",
        [[1, 1, 1, 0, -1], [1, 1, -1, -1, 0], [1, 1, 1, -1, -1]],
    )
    def test_below_threshold_is_flat(self, scores: list[int]) -> None:
        direction, _ = _confluence(scores, 4)
        assert direction is None

    def test_min_agree_five_is_unanimous_only(self) -> None:
        assert _confluence([1, 1, 1, 1, 1], 5) == (Direction.LONG, 5)
        assert _confluence([1, 1, 1, 1, 0], 5)[0] is None


class TestConstruction:
    @pytest.mark.parametrize("bad", [0, 6, -1])
    def test_min_agree_bounds(self, bad: int) -> None:
        with pytest.raises(ValueError, match="min_agree"):
            CrossAssetConfluenceStrategy("^OMX", min_agree=bad)

    def test_positive_windows(self) -> None:
        with pytest.raises(ValueError, match="zscore_window"):
            CrossAssetConfluenceStrategy("^OMX", zscore_window=0)


class TestEndToEnd:
    def test_unanimous_risk_on_goes_long(self) -> None:
        strat = CrossAssetConfluenceStrategy("^OMX", **_PARAMS)
        signals = list(strat.generate_signals(_risk_on_frame()))
        assert signals, "expected at least one signal once the shock lands"
        assert all(s.direction == Direction.LONG for s in signals)
        assert all(s.instrument == "^OMX" for s in signals)
        assert signals[-1].conviction == Conviction.VERY_HIGH  # all 5 agree
        assert signals[-1].suggested_target > signals[-1].suggested_entry

    def test_unanimous_risk_off_goes_short(self) -> None:
        strat = CrossAssetConfluenceStrategy("^OMX", **_PARAMS)
        signals = list(strat.generate_signals(_risk_off_frame()))
        assert signals
        assert all(s.direction == Direction.SHORT for s in signals)
        assert signals[-1].suggested_target < signals[-1].suggested_entry  # short

    def test_no_confluence_is_flat(self) -> None:
        # Only SPX is decisive; everything else dead flat -> never reaches 4/5.
        frame = _wide(
            spx=[100.0 + i for i in range(12)],
            vix=[20.0] * 12,
            tnx=[100.0] * 12,
            dxy=[100.0] * 12,
            gold=[100.0] * 12,
        )
        strat = CrossAssetConfluenceStrategy("^OMX", **_PARAMS)
        assert list(strat.generate_signals(frame)) == []

    def test_emitted_signals_meet_confluence_threshold(self) -> None:
        strat = CrossAssetConfluenceStrategy("^OMX", **_PARAMS)
        signals = list(strat.generate_signals(_risk_on_frame()))
        # confluence_factors lists only the agreeing signals -> must be >= min_agree.
        assert all(len(s.confluence_factors) >= 4 for s in signals)


class TestNoLookahead:
    def test_day_t_decision_ignores_day_t_data(self) -> None:
        """Poisoning a signal day's OWN cross-asset/OMX values must not change that
        day's signal — it may only affect the *next* day (the one-day shift)."""
        strat = CrossAssetConfluenceStrategy("^OMX", **_PARAMS)
        frame = _risk_on_frame()

        before = {s.timestamp: s for s in strat.generate_signals(frame)}
        assert before, "fixture must produce signals to test"

        signal_days = sorted(before)
        # Pick a signal day that has a following row, so propagation is observable.
        day_d = next(d for d in signal_days if d != frame.index[-1].to_pydatetime())
        pos = frame.index.get_loc(pd.Timestamp(day_d))

        poisoned = frame.copy()
        # Flip day D's own values to a violent risk-OFF extreme + wild OMX prices.
        poisoned.iloc[pos, poisoned.columns.get_loc("SPX_Close")] = 1.0
        poisoned.iloc[pos, poisoned.columns.get_loc("VIX_Close")] = 999.0
        poisoned.iloc[pos, poisoned.columns.get_loc("TNX_Close")] = 1.0
        poisoned.iloc[pos, poisoned.columns.get_loc("DXY_Close")] = 999.0
        poisoned.iloc[pos, poisoned.columns.get_loc("GOLD_Close")] = 999.0
        poisoned.iloc[pos, poisoned.columns.get_loc("Open")] = 1.0
        poisoned.iloc[pos, poisoned.columns.get_loc("Close")] = 1.0

        after = {s.timestamp: s for s in strat.generate_signals(poisoned)}

        # 1) Day D's decision is byte-for-byte identical -> no same-day leak.
        assert after.get(day_d) == before[day_d]
        # 2) But the poison DID propagate forward (proves we use the data, shifted).
        assert after != before
