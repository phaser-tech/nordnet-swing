"""Unit tests for the mean-reversion-after-extreme strategy. Fixtures only."""

from __future__ import annotations

import pandas as pd
import pytest

from packages.core.domain.signal import Direction
from packages.strategies.mean_reversion_extreme import (
    MeanReversionExtremeStrategy,
    bet_direction_series,
    extreme_zscores,
)

WINDOW = 5
SIGMA = 2.0
# A calm alternating +/-0.5% regime (|z| ~ 1, never triggers) so a single big
# move stands out cleanly. Returns are positions 1.. of the close series.
_CALM = [0.005, -0.005] * 5  # 10 returns -> warmup satisfied before position 11


def _frame_from_returns(returns: list[float], start: str = "2022-11-01") -> pd.DataFrame:
    """Build an OHLCV frame whose close-to-close returns equal `returns`.

    Open is set to the prior close (gap-free baseline); the strategy reads only
    Close, so Open/High/Low are deterministic padding here.
    """
    closes = [100.0]
    for r in returns:
        closes.append(closes[-1] * (1.0 + r))
    idx = pd.bdate_range(start, periods=len(closes), tz="UTC")
    opens = [closes[0], *closes[:-1]]
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=pd.DatetimeIndex(idx, name="ts"),
    )


class TestExtremeZscores:
    def test_big_drop_is_strongly_negative(self) -> None:
        df = _frame_from_returns([*_CALM, -0.05])
        z = extreme_zscores(df["Close"], WINDOW)
        assert z.iloc[-1] <= -SIGMA

    def test_big_jump_is_strongly_positive(self) -> None:
        df = _frame_from_returns([*_CALM, 0.05])
        z = extreme_zscores(df["Close"], WINDOW)
        assert z.iloc[-1] >= SIGMA

    def test_calm_day_is_within_band(self) -> None:
        df = _frame_from_returns(_CALM)
        z = extreme_zscores(df["Close"], WINDOW)
        # last calm day never reaches the 2-sigma band
        assert abs(z.iloc[-1]) < SIGMA


class TestBetDirection:
    def test_down_extreme_bets_long(self) -> None:
        df = _frame_from_returns([*_CALM, -0.05])
        bet = bet_direction_series(df, window=WINDOW, sigma=SIGMA)
        assert bet.iloc[-1] is Direction.LONG

    def test_up_extreme_bets_short(self) -> None:
        df = _frame_from_returns([*_CALM, 0.05])
        bet = bet_direction_series(df, window=WINDOW, sigma=SIGMA)
        assert bet.iloc[-1] is Direction.SHORT

    def test_calm_day_no_bet(self) -> None:
        df = _frame_from_returns(_CALM)
        bet = bet_direction_series(df, window=WINDOW, sigma=SIGMA)
        assert bet.iloc[-1] is None

    def test_index_matches_bars(self) -> None:
        df = _frame_from_returns([*_CALM, -0.05])
        bet = bet_direction_series(df, window=WINDOW, sigma=SIGMA)
        assert list(bet.index) == list(df.index)


class TestGenerateSignals:
    def _strategy(self) -> MeanReversionExtremeStrategy:
        return MeanReversionExtremeStrategy("TEST", window=WINDOW, sigma=SIGMA)

    def test_down_extreme_emits_long_on_next_day(self) -> None:
        # extreme at position 11, a calm T+1 bar follows at position 12
        df = _frame_from_returns([*_CALM, -0.05, 0.0])
        sigs = list(self._strategy().generate_signals(df))
        extreme_pos = 11
        expected_ts = df.index[extreme_pos + 1].to_pydatetime()
        matched = [s for s in sigs if s.timestamp == expected_ts]
        assert len(matched) == 1
        assert matched[0].direction == Direction.LONG
        assert matched[0].instrument == "TEST"

    def test_up_extreme_emits_short_on_next_day(self) -> None:
        df = _frame_from_returns([*_CALM, 0.05, 0.0])
        sigs = list(self._strategy().generate_signals(df))
        expected_ts = df.index[12].to_pydatetime()
        matched = [s for s in sigs if s.timestamp == expected_ts]
        assert len(matched) == 1
        assert matched[0].direction == Direction.SHORT

    def test_no_signal_emitted_for_the_last_bar(self) -> None:
        # extreme on the final bar cannot be traded (no T+1) -> no signal
        df = _frame_from_returns([*_CALM, -0.05])
        sigs = list(self._strategy().generate_signals(df))
        last_ts = df.index[-1].to_pydatetime()
        assert all(s.timestamp != last_ts for s in sigs)

    def test_extreme_inside_warmup_does_not_trigger(self) -> None:
        # a -5% move at position 1 has < window trailing returns -> z is NaN
        df = _frame_from_returns([-0.05, 0.01, -0.01, 0.01, -0.01])
        sigs = list(self._strategy().generate_signals(df))
        assert sigs == []

    def test_no_lookahead_decision_uses_only_through_day_T(self) -> None:
        df = _frame_from_returns([*_CALM, -0.05, 0.0, 0.01])
        strat = self._strategy()
        base = {s.timestamp: s.direction for s in strat.generate_signals(df)}

        # Mutate the T+1 bar's own OHLC drastically. The T+1-timestamped signal
        # is decided from closes <= T, so its direction must not change.
        extreme_pos = 11
        tplus1 = df.index[extreme_pos + 1]
        mutated = df.copy()
        mutated.loc[tplus1, ["Open", "High", "Low", "Close"]] = [50.0, 51.0, 49.0, 50.0]
        after = {s.timestamp: s.direction for s in strat.generate_signals(mutated)}

        ts = tplus1.to_pydatetime()
        assert ts in base and ts in after
        assert base[ts] == after[ts] == Direction.LONG


class TestValidation:
    @pytest.mark.parametrize("bad_window", [0, 1, -3])
    def test_rejects_small_window(self, bad_window: int) -> None:
        with pytest.raises(ValueError, match="window"):
            MeanReversionExtremeStrategy("TEST", window=bad_window, sigma=SIGMA)

    @pytest.mark.parametrize("bad_sigma", [0.0, -1.0])
    def test_rejects_nonpositive_sigma(self, bad_sigma: float) -> None:
        with pytest.raises(ValueError, match="sigma"):
            MeanReversionExtremeStrategy("TEST", window=WINDOW, sigma=bad_sigma)
