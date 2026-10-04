"""Tests for the metrics module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lionelmessi import metrics


def test_career_summary_values(sample_events: pl.DataFrame) -> None:
    summary = metrics.career_summary(sample_events)
    assert summary["matches"] == 1
    assert summary["minutes"] == 3
    assert summary["goals"] == 1
    assert summary["assists"] == 1
    assert summary["shots"] == 2
    assert summary["key_passes"] == 1
    assert summary["dribbles"] == 1
    assert summary["progressive_carries"] == 1
    assert summary["progressive_passes"] == 3
    assert summary["xg"] == pytest.approx(0.3)


def test_career_summary_empty_frame() -> None:
    summary = metrics.career_summary(pl.DataFrame())
    assert summary["matches"] == 0
    assert summary["goals"] == 0
    assert summary["xg"] == 0.0


def test_seasonal_breakdown(sample_events: pl.DataFrame) -> None:
    breakdown = metrics.seasonal_breakdown(sample_events)
    assert breakdown.height == 1
    row = breakdown.row(0, named=True)
    assert row["season_name"] == "2011/2012"
    assert row["goals"] == 1
    assert row["progressive_actions"] >= 3


def test_competition_breakdown(sample_events: pl.DataFrame) -> None:
    breakdown = metrics.competition_breakdown(sample_events)
    assert breakdown["competition_name"].to_list() == ["La Liga"]


def test_per_90(sample_events: pl.DataFrame) -> None:
    per90 = metrics.per_90(sample_events)
    assert "goals_per_90" in per90.columns
    assert per90["goals_per_90"][0] == pytest.approx(1 / 90.0, abs=5e-4)


def test_shot_and_assist_maps(sample_events: pl.DataFrame) -> None:
    shots = metrics.shot_map_data(sample_events)
    assert shots.height == 2
    assert set(["x", "y", "xg", "outcome", "is_goal"]).issubset(shots.columns)
    assists = metrics.assist_map_data(sample_events)
    assert assists.height == 1
    assert assists["recipient"][0] == "Sergio Busquets i Burgos"


def test_progressive_actions(sample_events: pl.DataFrame) -> None:
    actions = metrics.progressive_actions(sample_events)
    assert actions.height == 4
    assert set(actions["type"].unique().to_list()).issubset({"Pass", "Carry"})


def test_pitch_zone_heatmap_shapes(sample_events: pl.DataFrame) -> None:
    thirds = metrics.pitch_zone_heatmap(sample_events, zone="thirds")
    assert thirds.shape == (4, 3)
    box = metrics.pitch_zone_heatmap(sample_events, zone="18")
    assert box.shape == (2, 1)
    custom = metrics.pitch_zone_heatmap(sample_events, zone="custom", bins=(8, 6))
    assert custom.shape == (6, 8)
    assert custom.sum() == 9  # all Messi touches
    assert np.asarray(custom).sum() > 0


def test_goal_breakdown(sample_events: pl.DataFrame) -> None:
    breakdown = metrics.goal_breakdown(sample_events, "evt-005")
    assert breakdown["scorer"] == "Lionel Andrés Messi Cuccittini"
    assert breakdown["n_events"] == 3
    assert breakdown["xg"] == pytest.approx(0.2)
    assert breakdown["actors"][0] == "Lionel Andrés Messi Cuccittini"

    assisted = metrics.goal_breakdown(sample_events, "evt-001")
    assert assisted["n_events"] == 2
    assert assisted["scorer"] == "Sergio Busquets i Burgos"  # the shot taker


def test_goal_breakdown_spec_keys(sample_events: pl.DataFrame) -> None:
    breakdown = metrics.goal_breakdown(sample_events, "evt-005")
    for key in (
        "goal_event",
        "chain_events",
        "start_xy",
        "end_xy",
        "duration_seconds",
        "actors",
        "xg",
    ):
        assert key in breakdown
    assert breakdown["goal_event"]["id"] == "evt-005"
    assert len(breakdown["chain_events"]) == 3
    assert breakdown["start_xy"] == (60.0, 20.0)
    assert breakdown["end_xy"] == (88.0, 30.0)
    assert breakdown["duration_seconds"] == pytest.approx(6.0)


def test_goal_breakdown_missing_id_raises(sample_events: pl.DataFrame) -> None:
    with pytest.raises(KeyError):
        metrics.goal_breakdown(sample_events, "nope")


def test_assist_breakdown(sample_events: pl.DataFrame) -> None:
    breakdown = metrics.assist_breakdown(sample_events, "evt-001")
    assert breakdown["assister"] == "Lionel Andrés Messi Cuccittini"
    assert breakdown["recipient"] == "Sergio Busquets i Burgos"
    with pytest.raises(KeyError):
        metrics.assist_breakdown(sample_events, "missing")


def test_rolling_form(sample_events: pl.DataFrame) -> None:
    form = metrics.rolling_form(sample_events, window=5)
    assert form.height == 1
    assert form["goal_contributions"][0] == 2  # one goal + one assist
    assert "rolling_5_contributions" in form.columns


def test_messi_only_helper(sample_events: pl.DataFrame) -> None:
    assert metrics.messi_only(sample_events).height == 9
    assert metrics.career_summary(sample_events, player_id=None)["goals"] == 2
