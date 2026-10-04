"""Tests for the action-value models."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lionelmessi import models


def test_xt_surface_shape(sample_events: pl.DataFrame) -> None:
    model = models.ExpectedThreat(x_bins=16, y_bins=12).fit(sample_events)
    assert model.xt_surface.shape == (12, 16)
    assert model.n_events > 0


def test_xt_values_decrease_away_from_goal(sample_events: pl.DataFrame) -> None:
    model = models.ExpectedThreat().fit(sample_events)
    near = model.xt_at(110.0, 40.0)
    far = model.xt_at(30.0, 40.0)
    assert isinstance(near, float)
    assert near > far


def test_xt_delta_matches_difference(sample_events: pl.DataFrame) -> None:
    model = models.ExpectedThreat().fit(sample_events)
    start, end = (60.0, 20.0), (110.0, 40.0)
    assert model.xt(start, end) == pytest.approx(model.xt_at(*end) - model.xt_at(*start))


def test_xt_unfitted_returns_zero(sample_events: pl.DataFrame) -> None:
    assert models.ExpectedThreat().xt_at(60.0, 40.0) == 0.0


def test_transition_rows_are_distributions(sample_events: pl.DataFrame) -> None:
    model = models.ExpectedThreat().fit(sample_events)
    row_sums = model.transition.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_value_iteration_converges(sample_events: pl.DataFrame) -> None:
    strict = models.ExpectedThreat().fit(sample_events, tol=1e-9, max_iter=500)
    repeat = models.ExpectedThreat().fit(sample_events, tol=1e-9, max_iter=500)
    assert np.allclose(strict.values, repeat.values)
    assert np.all(np.isfinite(strict.values))
    assert np.all(strict.values >= 0.0)
    assert float(strict.values.max()) <= 1.0 + 1e-9


def test_save_and_load_roundtrip(sample_events: pl.DataFrame, tmp_path) -> None:
    model = models.ExpectedThreat().fit(sample_events)
    path = tmp_path / "xt.npz"
    model.save(path)
    assert path.exists()
    loaded = models.ExpectedThreat.load(path)
    assert loaded.x_bins == model.x_bins
    assert np.allclose(loaded.values, model.values)
    assert loaded.xt_at(110.0, 40.0) == pytest.approx(model.xt_at(110.0, 40.0))


def test_action_value_model(sample_events: pl.DataFrame) -> None:
    model = models.ActionValueModel()
    model.fit(sample_events)
    scores = model.score(sample_events)
    assert scores.shape[0] == sample_events.height
    assert np.all((scores >= 0) & (scores <= 1))


def test_pitch_control_surface() -> None:
    surface = models.pitch_control_surface((60.0, 40.0))
    assert surface.shape == (12, 16)
    assert surface.max() <= 1.0
    assert surface.min() >= 0.0
    # Control decays with distance from the ball.
    assert surface[5, 7] > surface[0, 0]


def _monotone_rows() -> list[dict[str, object]]:
    """Craft data where threat is exactly 0 in cells 0-2 and 1 in cells 3-15.

    Along the y=40 row (y-bin 6): cells 0-2 only pass backwards within the
    zero-value region, cells 3-14 chain forward, and cell 15 shoots and always
    scores.
    """

    rows: list[dict[str, object]] = []

    def move(sx: float, sy: float, ex: float, ey: float) -> None:
        rows.append(
            {
                "location_x": sx,
                "location_y": sy,
                "is_shot": False,
                "is_goal": False,
                "is_pass": True,
                "is_carry": False,
                "pass_end_x": ex,
                "pass_end_y": ey,
            }
        )

    def shot(sx: float, sy: float, scored: bool) -> None:
        rows.append(
            {
                "location_x": sx,
                "location_y": sy,
                "is_shot": True,
                "is_goal": scored,
                "is_pass": False,
                "is_carry": False,
                "pass_end_x": None,
                "pass_end_y": None,
            }
        )

    # Backward-only moves: cells 0, 1, 2 stay confined to the zero-value region.
    move(3.0, 42.0, 3.0, 41.0)
    move(10.0, 42.0, 3.0, 42.0)
    move(18.0, 42.0, 10.0, 42.0)
    # Forward chain from cell 3 up to the shooting cell 15.
    for xi in range(3, 15):
        move(xi * 7.5 + 3.0, 42.0, (xi + 1) * 7.5 + 3.0, 42.0)
    # Cell 15: every action is a shot and every shot scores.
    for _ in range(3):
        shot(118.0, 42.0, True)
    return rows


def test_xt_boundary_invariants() -> None:
    frame = pl.DataFrame(_monotone_rows())
    model = models.ExpectedThreat(x_bins=16, y_bins=12).fit(frame)
    assert model.xt_at(120.0, 40.0) == pytest.approx(1.0, abs=1e-6)
    assert model.xt_at(0.0, 40.0) == 0.0


def test_xt_monotone_along_centre_row() -> None:
    frame = pl.DataFrame(_monotone_rows())
    model = models.ExpectedThreat().fit(frame)
    values = [model.xt_at(xi * 7.5 + 3.75, 40.0) for xi in range(16)]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))
    assert values[0] == 0.0
    assert values[-1] == pytest.approx(1.0, abs=1e-6)


def test_xt_fit_array_form_matches_frame() -> None:
    rows = _monotone_rows()
    frame_model = models.ExpectedThreat().fit(pl.DataFrame(rows))

    x = [float(r["location_x"]) for r in rows]  # type: ignore[arg-type]
    y = [float(r["location_y"]) for r in rows]  # type: ignore[arg-type]
    actions = ["shot" if r["is_shot"] else "move" for r in rows]
    goals = [bool(r["is_goal"]) for r in rows]
    end_x = [float(r["pass_end_x"]) if r["pass_end_x"] is not None else 0.0 for r in rows]  # type: ignore[arg-type]
    end_y = [float(r["pass_end_y"]) if r["pass_end_y"] is not None else 0.0 for r in rows]  # type: ignore[arg-type]

    array_model = models.ExpectedThreat().fit(x, y, actions, goals, end_x=end_x, end_y=end_y)
    assert array_model.n_events == frame_model.n_events
    assert np.allclose(array_model.values, frame_model.values)
    assert array_model.xt_at(120.0, 40.0) == pytest.approx(1.0, abs=1e-6)
    assert array_model.xt_at(0.0, 40.0) == 0.0


def test_xt_fit_array_form_validation() -> None:
    with pytest.raises(ValueError):
        models.ExpectedThreat().fit([1.0, 2.0], [3.0], ["move", "move"], [False, False])
    with pytest.raises(ValueError):
        models.ExpectedThreat().fit([1.0], [3.0], ["tackle"], [False])
    with pytest.raises(ValueError):
        models.ExpectedThreat().fit([1.0], [3.0], ["move"], [False], end_x=[2.0])
    with pytest.raises(TypeError):
        models.ExpectedThreat().fit([1.0, 2.0])  # type: ignore[call-overload]
