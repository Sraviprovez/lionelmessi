"""Action-value models.

The centrepiece is :class:`ExpectedThreat`, a grid-based xT model in the
tradition of Karun Singh's expected threat.  A transition matrix is estimated
from *observed* pass and carry end locations rather than a uniform prior, and
cell values are obtained by value iteration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt
import polars as pl

from lionelmessi import config

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

__all__ = ["ExpectedThreat", "ActionValueModel", "pitch_control_surface"]

#: Array-like inputs accepted by the array form of :meth:`ExpectedThreat.fit`.
Labels = Sequence[str] | npt.NDArray[np.str_]
FlagArray = Sequence[bool] | npt.NDArray[np.bool_]
PointArray = Sequence[float] | npt.NDArray[np.float64]

#: One normalized observation: (start_x, start_y, kind, scored, end_x, end_y).
_Observation = tuple[float, float, str, bool, float | None, float | None]


def _cell_index(x: float, y: float, x_bins: int, y_bins: int) -> int:
    xi = int(np.clip(np.floor(x / (config.PITCH_LENGTH / x_bins)), 0, x_bins - 1))
    yi = int(np.clip(np.floor(y / (config.PITCH_WIDTH / y_bins)), 0, y_bins - 1))
    return xi * y_bins + yi


def _observations_from_frame(frame: pl.DataFrame, player_id: int | None) -> list[_Observation]:
    """Normalize an event frame into xT observations."""

    if player_id is not None:
        frame = frame.filter(pl.col("player_id") == player_id)
    observations: list[_Observation] = []
    for row in frame.iter_rows(named=True):
        start_x = row.get("location_x")
        start_y = row.get("location_y")
        if start_x is None or start_y is None:
            continue
        if bool(row.get("is_shot")):
            observations.append(
                (float(start_x), float(start_y), "shot", bool(row.get("is_goal")), None, None)
            )
        elif bool(row.get("is_pass")) or bool(row.get("is_carry")):
            end_x = row.get("pass_end_x")
            end_y = row.get("pass_end_y")
            if end_x is None or end_y is None:
                end_x = row.get("carry_end_x")
                end_y = row.get("carry_end_y")
            if end_x is None or end_y is None:
                continue
            observations.append(
                (float(start_x), float(start_y), "move", False, float(end_x), float(end_y))
            )
    return observations


def _observations_from_arrays(
    x: PointArray,
    y: PointArray,
    actions: Labels,
    goals: FlagArray,
    end_x: PointArray | None,
    end_y: PointArray | None,
) -> list[_Observation]:
    """Normalize parallel arrays into xT observations."""

    n = len(x)
    if len(y) != n or len(actions) != n or len(goals) != n:
        raise ValueError("x, y, actions, and goals must have the same length")
    if (end_x is None) != (end_y is None):
        raise ValueError("end_x and end_y must be provided together")
    observations: list[_Observation] = []
    for i in range(n):
        kind = str(actions[i]).strip().lower()
        if kind not in ("shot", "move"):
            raise ValueError(f"Unknown action {actions[i]!r}; expected 'shot' or 'move'")
        finish_x = float(end_x[i]) if end_x is not None else None
        finish_y = float(end_y[i]) if end_y is not None else None
        observations.append((float(x[i]), float(y[i]), kind, bool(goals[i]), finish_x, finish_y))
    return observations


@dataclass
class ExpectedThreat:
    """A grid-based Expected Threat model.

    Parameters
    ----------
    x_bins, y_bins:
        Grid resolution along the length and width of the pitch.
    """

    x_bins: int = 16
    y_bins: int = 12
    p_shot: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    p_goal: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    transition: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    values: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    n_events: int = 0

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    @property
    def n_cells(self) -> int:
        return self.x_bins * self.y_bins

    @overload
    def fit(
        self,
        events: pl.DataFrame,
        *,
        tol: float = ...,
        max_iter: int = ...,
        player_id: int | None = ...,
    ) -> ExpectedThreat: ...

    @overload
    def fit(
        self,
        x: PointArray,
        y: PointArray,
        actions: Labels,
        goals: FlagArray,
        /,
        *,
        end_x: PointArray | None = ...,
        end_y: PointArray | None = ...,
        tol: float = ...,
        max_iter: int = ...,
    ) -> ExpectedThreat: ...

    def fit(
        self,
        events: pl.DataFrame | PointArray,
        y: PointArray | None = None,
        actions: Labels | None = None,
        goals: FlagArray | None = None,
        *,
        end_x: PointArray | None = None,
        end_y: PointArray | None = None,
        tol: float = 1e-6,
        max_iter: int = 100,
        player_id: int | None = None,
    ) -> ExpectedThreat:
        """Estimate the model from observed shot, pass, and carry data.

        Two calling conventions are supported:

        * ``fit(events)`` — a Polars event frame with the derived boolean
          columns produced by :func:`lionelmessi.data.add_derived_columns`.
        * ``fit(x, y, actions, goals, end_x=..., end_y=...)`` — parallel
          arrays of start coordinates, action kinds (``"shot"`` or
          ``"move"``), whether the shot scored, and optional move end
          coordinates used to estimate the transition matrix.
        """

        if isinstance(events, pl.DataFrame):
            observations = _observations_from_frame(events, player_id)
        else:
            if y is None or actions is None or goals is None:
                raise TypeError("fit(x, y, actions, goals) requires all four arrays")
            observations = _observations_from_arrays(events, y, actions, goals, end_x, end_y)

        n = self.n_cells
        shot_counts = np.zeros(n)
        goal_counts = np.zeros(n)
        move_counts = np.zeros((n, n))
        move_totals = np.zeros(n)

        for start_x, start_y, kind, scored, finish_x, finish_y in observations:
            start = _cell_index(start_x, start_y, self.x_bins, self.y_bins)
            if kind == "shot":
                shot_counts[start] += 1
                if scored:
                    goal_counts[start] += 1
            elif finish_x is not None and finish_y is not None:
                end = _cell_index(finish_x, finish_y, self.x_bins, self.y_bins)
                move_counts[start, end] += 1
                move_totals[start] += 1

        self.n_events = len(observations)
        total = shot_counts + move_totals
        with np.errstate(divide="ignore", invalid="ignore"):
            self.p_shot = np.where(total > 0, shot_counts / total, 0.0)
            self.p_goal = np.where(shot_counts > 0, goal_counts / shot_counts, 0.0)

        transition = np.zeros((n, n))
        for i in range(n):
            if move_totals[i] > 0:
                transition[i] = move_counts[i] / move_totals[i]
            else:
                transition[i] = np.full(n, 1.0 / n)
        self.transition = transition

        self.values = self._value_iteration(tol=tol, max_iter=max_iter)
        return self

    def _value_iteration(self, *, tol: float, max_iter: int) -> npt.NDArray[np.float64]:
        values: npt.NDArray[np.float64] = np.zeros(self.n_cells)
        p_move = 1.0 - self.p_shot
        for _ in range(max_iter):
            updated = self.p_shot * self.p_goal + p_move * (self.transition @ values)
            if float(np.max(np.abs(updated - values))) < tol:
                values = updated
                break
            values = updated
        return values

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #
    @property
    def xt_surface(self) -> npt.NDArray[np.float64]:
        """Return the xT surface with shape ``(y_bins, x_bins)``."""

        grid = self.values.reshape(self.x_bins, self.y_bins)
        return np.asarray(grid.T)

    def xt_at(self, x: float, y: float) -> float:
        """Return the xT value at pitch coordinate ``(x, y)``."""

        if self.values.size == 0:
            return 0.0
        return float(self.values[_cell_index(x, y, self.x_bins, self.y_bins)])

    def xt(self, start_xy: tuple[float, float], end_xy: tuple[float, float]) -> float:
        """Return the xT gained moving from ``start_xy`` to ``end_xy``."""

        return self.xt_at(*end_xy) - self.xt_at(*start_xy)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        """Persist the model to a ``.npz`` archive."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            x_bins=self.x_bins,
            y_bins=self.y_bins,
            p_shot=self.p_shot,
            p_goal=self.p_goal,
            transition=self.transition,
            values=self.values,
            n_events=self.n_events,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> ExpectedThreat:
        """Load a model previously written by :meth:`save`."""

        with np.load(Path(path)) as data:
            return cls(
                x_bins=int(data["x_bins"]),
                y_bins=int(data["y_bins"]),
                p_shot=data["p_shot"],
                p_goal=data["p_goal"],
                transition=data["transition"],
                values=data["values"],
                n_events=int(data["n_events"]),
            )

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the xT surface via :mod:`lionelmessi.viz`."""

        from lionelmessi import viz

        return viz.plot_xt_surface(self, ax=ax, **kwargs)


class ActionValueModel:
    """A VAEP-style action-value model (requires the ``ml`` extra).

    The model learns ``P(goal | frame)`` from frame-level features and scores
    each action by the change in that probability.
    """

    def __init__(self) -> None:
        self._model: Any | None = None

    @staticmethod
    def frame_features(events: pl.DataFrame) -> npt.NDArray[np.float64]:
        """Build the numeric feature matrix used by the classifier."""

        rows = []
        for row in events.iter_rows(named=True):
            x = row.get("location_x")
            y = row.get("location_y")
            if x is None or y is None:
                continue
            dist = float(np.hypot(config.PITCH_LENGTH - x, config.PITCH_WIDTH / 2 - y))
            angle = float(np.arctan2(abs(y - config.PITCH_WIDTH / 2), config.PITCH_LENGTH - x))
            rows.append(
                [
                    x,
                    y,
                    dist,
                    angle,
                    1.0 if row.get("under_pressure") else 0.0,
                    1.0 if row.get("is_shot") else 0.0,
                ]
            )
        return np.asarray(rows, dtype=float)

    def fit(self, events: pl.DataFrame) -> ActionValueModel:
        """Fit the underlying classifier on labelled events."""

        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        features = self.frame_features(events)
        labels = np.asarray(
            [1 if row.get("is_goal") else 0 for row in events.iter_rows(named=True)],
            dtype=int,
        )
        if features.size == 0 or labels.size == 0:
            raise ValueError("No usable rows to fit the action-value model")
        model = LogisticRegression(max_iter=500)
        model.fit(features, labels)
        self._model = model
        return self

    def score(self, events: pl.DataFrame) -> npt.NDArray[np.float64]:
        """Return ``P(goal | state)`` for each gem in ``events``."""

        if self._model is None:
            raise RuntimeError("Call fit() before score()")
        features = self.frame_features(events)
        return np.asarray(self._model.predict_proba(features)[:, 1])


def pitch_control_surface(
    ball_xy: tuple[float, float],
    *,
    x_bins: int = 16,
    y_bins: int = 12,
    time_to_intercept: float = 0.7,
) -> npt.NDArray[np.float64]:
    """A simple ball-relative pitch-control surface.

    Control at a cell decays with the time it would take the ball to travel
    there, approximated from Euclidean distance.
    """

    xs = (np.arange(x_bins) + 0.5) * (config.PITCH_LENGTH / x_bins)
    ys = (np.arange(y_bins) + 0.5) * (config.PITCH_WIDTH / y_bins)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    distance = np.hypot(grid_x - ball_xy[0], grid_y - ball_xy[1])
    travel_time = distance / 15.0
    return np.asarray(np.exp(-np.maximum(travel_time - time_to_intercept, 0.0)), dtype=float)
