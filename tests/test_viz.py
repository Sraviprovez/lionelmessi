"""Smoke tests for the visualization module."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest
from matplotlib.axes import Axes

from lionelmessi import chains, metrics, models, viz


@pytest.fixture()
def xt_model(sample_events: pl.DataFrame) -> models.ExpectedThreat:
    return models.ExpectedThreat().fit(sample_events)


def test_all_plots_return_axes(
    sample_events: pl.DataFrame, xt_model: models.ExpectedThreat
) -> None:
    goals = sample_events.filter(pl.col("is_goal"))
    built_chain = chains.build_continuations(sample_events, xt_model=xt_model)[0]
    plots: list[tuple[str, Callable[[], Axes]]] = [
        ("pitch", lambda: viz.plot_pitch()),
        ("xt", lambda: viz.plot_xt_surface(xt_model, annotate=True)),
        ("shot", lambda: viz.plot_shot_map(metrics.shot_map_data(sample_events))),
        ("shot_raw", lambda: viz.plot_shot_map(sample_events)),
        ("goal", lambda: viz.plot_goal_map(goals)),
        ("assist", lambda: viz.plot_assist_map(sample_events)),
        ("pass", lambda: viz.plot_pass_map(sample_events)),
        ("pass_prog", lambda: viz.plot_pass_map(sample_events, progressive_only=True)),
        ("network", lambda: viz.plot_pass_network(sample_events)),
        ("carry", lambda: viz.plot_carry_map(sample_events)),
        ("heat", lambda: viz.plot_heatmap(sample_events)),
        ("season_heat", lambda: viz.plot_season_heatmap(sample_events)),
        ("continuation", lambda: viz.plot_continuation(built_chain)),
        ("goal_breakdown", lambda: viz.plot_goal_breakdown(sample_events, "evt-005")),
        ("timeline", lambda: viz.plot_career_timeline(sample_events)),
        ("form", lambda: viz.plot_rolling_form(sample_events, window=3)),
    ]
    for name, fn in plots:
        ax = fn()
        assert isinstance(ax, Axes), name


def test_save_plot_writes_file(sample_events: pl.DataFrame, tmp_path) -> None:
    ax = viz.plot_shot_map(metrics.shot_map_data(sample_events))
    path = viz.save_plot(ax, tmp_path / "shots.png", dpi=80)
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_pitch_half(sample_events: pl.DataFrame) -> None:
    ax = viz.plot_pitch(half=True)
    assert ax.get_xlim()[1] == 60.0


def test_plot_pitch_prefers_mplsoccer_when_available() -> None:
    pitch_cls = viz._mplsoccer_pitch()
    if pitch_cls is None:  # pragma: no cover - viz extra not installed
        pytest.skip("mplsoccer not installed")
    ax = viz.plot_pitch()
    # mplsoccer pads the pitch by a few metres on each side.
    assert ax.get_xlim()[1] > 120.0


def test_plot_pitch_native_fallback(
    sample_events: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(viz, "_mplsoccer_pitch", lambda: None)
    ax = viz.plot_pitch()
    assert ax.get_xlim() == (0.0, 120.0)
    assert ax.get_ylim() == (0.0, 80.0)
    ax = viz.plot_shot_map(metrics.shot_map_data(sample_events))
    assert isinstance(ax, Axes)
