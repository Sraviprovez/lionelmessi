"""Career metrics, aggregations, and breakdowns.

Every function is a pure function of its input frame.  Action counts are
attributed to Messi (``player_id == 5503``) by default; pass ``player_id=None``
to aggregate over every player in the supplied frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from lionelmessi import config

__all__ = [
    "career_summary",
    "seasonal_breakdown",
    "competition_breakdown",
    "per_90",
    "shot_map_data",
    "assist_map_data",
    "progressive_actions",
    "pitch_zone_heatmap",
    "goal_breakdown",
    "assist_breakdown",
    "rolling_form",
    "messi_only",
]

MESSI_ID = config.MESSI_PLAYER_ID


def _player_filter(frame: pl.DataFrame, player_id: int | None) -> pl.DataFrame:
    if player_id is None:
        return frame
    return frame.filter(pl.col("player_id") == player_id)


def messi_only(frame: pl.DataFrame) -> pl.DataFrame:
    """Return only Messi's own actions."""

    return _player_filter(frame, MESSI_ID)


def _match_minutes(frame: pl.DataFrame) -> int:
    """Approximate minutes played per match, summed across the frame.

    This is a pragmatic proxy: we take the furthest minute reached in each
    match (regulation time) rather than reconstructing substitutions.
    """

    if not frame.height:
        return 0
    per_match = (
        frame.group_by("match_id")
        .agg(pl.col("minute").max().alias("last_minute"))
        .with_columns(
            pl.when(pl.col("last_minute") > 90)
            .then(90)
            .otherwise(pl.col("last_minute"))
            .alias("minutes")
        )
    )
    return int(per_match["minutes"].fill_null(0).sum())


def _sum_flag(frame: pl.DataFrame, flag: str, player_id: int | None = MESSI_ID) -> int:
    sub = _player_filter(frame, player_id)
    if not sub.height or flag not in sub.columns:
        return 0
    return int(sub[flag].fill_null(False).sum())


def career_summary(frame: pl.DataFrame, *, player_id: int | None = MESSI_ID) -> dict[str, Any]:
    """Return a headline career summary dictionary."""

    if not frame.height:
        return {
            "matches": 0,
            "minutes": 0,
            "goals": 0,
            "assists": 0,
            "shots": 0,
            "xg": 0.0,
            "key_passes": 0,
            "dribbles": 0,
            "progressive_carries": 0,
            "progressive_passes": 0,
        }

    mine = _player_filter(frame, player_id)
    goals = _sum_flag(mine, "is_goal", None)
    assists = _sum_flag(mine, "is_assist", None)
    shots = _sum_flag(mine, "is_shot", None)
    key_passes = _sum_flag(mine, "is_key_pass", None)
    prog_carries = (
        int(mine.filter(pl.col("is_carry"))["is_progressive"].fill_null(False).sum())
        if mine.height
        else 0
    )
    prog_passes = (
        int(mine.filter(pl.col("is_pass"))["is_progressive"].fill_null(False).sum())
        if mine.height
        else 0
    )
    dribbles = (
        int(
            mine.filter(
                (pl.col("type") == "Dribble") & (pl.col("dribble_outcome") == "Complete")
            ).height
        )
        if mine.height
        else 0
    )
    xg = float(mine.filter(pl.col("is_shot"))["shot_xg"].fill_null(0.0).sum()) if shots else 0.0

    return {
        "matches": int(frame["match_id"].n_unique()) if frame.height else 0,
        "minutes": _match_minutes(frame),
        "goals": goals,
        "assists": assists,
        "shots": shots,
        "xg": round(xg, 3),
        "key_passes": key_passes,
        "dribbles": dribbles,
        "progressive_carries": prog_carries,
        "progressive_passes": prog_passes,
    }


def _aggregate(
    frame: pl.DataFrame, keys: str | list[str], *, player_id: int | None = MESSI_ID
) -> pl.DataFrame:
    mine = _player_filter(frame, player_id)
    keys = [keys] if isinstance(keys, str) else keys
    if not mine.height:
        return pl.DataFrame()
    aggregated = mine.group_by(keys).agg(
        [
            pl.col("match_id").n_unique().alias("matches"),
            pl.col("is_goal").fill_null(False).sum().alias("goals"),
            pl.col("is_assist").fill_null(False).sum().alias("assists"),
            pl.col("is_shot").fill_null(False).sum().alias("shots"),
            pl.col("shot_xg").fill_null(0.0).sum().alias("xg"),
            pl.col("is_key_pass").fill_null(False).sum().alias("key_passes"),
            pl.col("is_progressive").fill_null(False).sum().alias("progressive_actions"),
        ]
    )
    return aggregated.sort(keys)


def seasonal_breakdown(frame: pl.DataFrame, *, player_id: int | None = MESSI_ID) -> pl.DataFrame:
    """Break Messi's output down by season."""

    return _aggregate(frame, "season_name", player_id=player_id)


def competition_breakdown(frame: pl.DataFrame, *, player_id: int | None = MESSI_ID) -> pl.DataFrame:
    """Break Messi's output down by competition."""

    return _aggregate(frame, "competition_name", player_id=player_id)


def per_90(frame: pl.DataFrame, *, player_id: int | None = MESSI_ID) -> pl.DataFrame:
    """Per-90 rates by season."""

    breakdown = seasonal_breakdown(frame, player_id=player_id)
    if not breakdown.height:
        return breakdown
    minutes_per_match = 90.0
    nineties = (breakdown["matches"].cast(pl.Float64) * minutes_per_match).clip(1.0)
    return breakdown.with_columns(
        [
            (pl.col("goals") / nineties).round(4).alias("goals_per_90"),
            (pl.col("assists") / nineties).round(4).alias("assists_per_90"),
            (pl.col("xg") / nineties).round(4).alias("xg_per_90"),
        ]
    )


def shot_map_data(frame: pl.DataFrame, *, player_id: int | None = MESSI_ID) -> pl.DataFrame:
    """Return one row per shot with coordinates, xG, and outcome."""

    mine = _player_filter(frame, player_id)
    shots = mine.filter(pl.col("is_shot"))
    if not shots.height:
        return pl.DataFrame(
            schema={
                "id": pl.Utf8,
                "match_id": pl.Int64,
                "season_name": pl.Utf8,
                "location_x": pl.Float64,
                "location_y": pl.Float64,
                "shot_xg": pl.Float64,
                "shot_outcome": pl.Utf8,
                "is_goal": pl.Boolean,
            }
        )
    return shots.select(
        [
            "id",
            "match_id",
            "season_name",
            pl.col("location_x").alias("x"),
            pl.col("location_y").alias("y"),
            pl.col("shot_xg").alias("xg"),
            pl.col("shot_outcome").alias("outcome"),
            "is_goal",
        ]
    )


def assist_map_data(frame: pl.DataFrame, *, player_id: int | None = MESSI_ID) -> pl.DataFrame:
    """Return one row per Messi assist with the pass origin."""

    mine = _player_filter(frame, player_id)
    assists = mine.filter(pl.col("is_assist"))
    if not assists.height:
        return pl.DataFrame(
            schema={
                "id": pl.Utf8,
                "match_id": pl.Int64,
                "season_name": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
                "end_x": pl.Float64,
                "end_y": pl.Float64,
                "recipient": pl.Utf8,
            }
        )
    return assists.select(
        [
            "id",
            "match_id",
            "season_name",
            pl.col("location_x").alias("x"),
            pl.col("location_y").alias("y"),
            pl.col("pass_end_x").alias("end_x"),
            pl.col("pass_end_y").alias("end_y"),
            pl.col("pass_recipient").alias("recipient"),
        ]
    )


def progressive_actions(frame: pl.DataFrame, *, player_id: int | None = MESSI_ID) -> pl.DataFrame:
    """Return Messi's progressive passes and carries."""

    mine = _player_filter(frame, player_id)
    progressive = mine.filter(pl.col("is_progressive") & (pl.col("is_pass") | pl.col("is_carry")))
    if not progressive.height:
        return pl.DataFrame(schema={"id": pl.Utf8, "match_id": pl.Int64, "type": pl.Utf8})
    return progressive.select(
        [
            "id",
            "match_id",
            "season_name",
            "type",
            "minute",
            pl.col("location_x").alias("x"),
            pl.col("location_y").alias("y"),
            pl.coalesce([pl.col("pass_end_x"), pl.col("carry_end_x")]).alias("end_x"),
            pl.coalesce([pl.col("pass_end_y"), pl.col("carry_end_y")]).alias("end_y"),
        ]
    )


def pitch_zone_heatmap(
    frame: pl.DataFrame,
    *,
    zone: str = "thirds",
    bins: tuple[int, int] = (16, 12),
    player_id: int | None = MESSI_ID,
) -> np.ndarray:
    """Return a 2D histogram of Messi's touches over the pitch.

    Parameters
    ----------
    zone:
        ``"thirds"`` or ``"18"`` collapse the histogram into thirds or into the
        18-yard box versus the rest; ``"custom"`` uses ``bins`` directly.
    bins:
        ``(x_bins, y_bins)`` grid for the ``"custom"`` mode.
    """

    mine = _player_filter(frame, player_id)
    points = mine.drop_nulls(["location_x", "location_y"]).select(["location_x", "location_y"])
    xs = points["location_x"].to_numpy()
    ys = points["location_y"].to_numpy()
    if xs.size == 0:
        return np.zeros((bins[1], bins[0]))

    if zone == "thirds":
        x_edges = np.array([0.0, 40.0, 80.0, config.PITCH_LENGTH])
        y_edges = np.array([0.0, 20.0, 40.0, 60.0, config.PITCH_WIDTH])
        hist, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
        return np.asarray(hist.T, dtype=float)
    if zone == "18":
        in_box = (xs >= 102) & (ys >= 18) & (ys <= 62)
        third_hist = np.array([[int((~in_box).sum())], [int(in_box.sum())]], dtype=float)
        return np.asarray(third_hist)
    x_edges = np.linspace(0, config.PITCH_LENGTH, bins[0] + 1)
    y_edges = np.linspace(0, config.PITCH_WIDTH, bins[1] + 1)
    hist, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    return np.asarray(hist.T, dtype=float)


def _find_event(frame: pl.DataFrame, event_id: str) -> pl.DataFrame | None:
    matches = frame.filter(pl.col("id") == event_id)
    if not matches.height:
        return None
    return matches


def _possession_chain(frame: pl.DataFrame, match_id: int, possession: int) -> pl.DataFrame:
    chain = frame.filter(
        (pl.col("match_id") == match_id) & (pl.col("possession") == possession)
    ).sort(["period", "index"])

    def _row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row.get("id"),
            "index": row.get("index"),
            "minute": row.get("minute"),
            "second": row.get("second"),
            "type": row.get("type"),
            "player": row.get("player"),
            "team": row.get("team"),
            "x": row.get("location_x"),
            "y": row.get("location_y"),
            "end_x": row.get("pass_end_x") or row.get("carry_end_x"),
            "end_y": row.get("pass_end_y") or row.get("carry_end_y"),
        }

    return pl.DataFrame([_row(r) for r in chain.iter_rows(named=True)])


def _chain_bounds(chain_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return start/end coordinates and duration for a chain's event rows."""

    start_xy: tuple[float, float] | None = None
    end_xy: tuple[float, float] | None = None
    for row in chain_rows:
        if row.get("x") is not None and row.get("y") is not None:
            if start_xy is None:
                start_xy = (float(row["x"]), float(row["y"]))
            end_xy = (float(row["x"]), float(row["y"]))
        if row.get("end_x") is not None and row.get("end_y") is not None:
            end_xy = (float(row["end_x"]), float(row["end_y"]))
    duration = 0.0
    if chain_rows:
        first = chain_rows[0]
        last = chain_rows[-1]
        first_seconds = (first.get("minute") or 0) * 60 + (first.get("second") or 0)
        last_seconds = (last.get("minute") or 0) * 60 + (last.get("second") or 0)
        duration = max(0.0, float(last_seconds - first_seconds))
    return {"start_xy": start_xy, "end_xy": end_xy, "duration_seconds": duration}


def goal_breakdown(frame: pl.DataFrame, goal_event_id: str) -> dict[str, Any]:
    """Reconstruct the possession chain that produced a goal.

    Returns a dict with the keys ``goal_event``, ``chain_events``, ``start_xy``,
    ``end_xy``, ``duration_seconds``, ``actors``, and ``xg`` (plus convenience
    aliases such as ``scorer``, ``minute``, and ``chain``).
    """

    goal_rows = _find_event(frame, goal_event_id)
    if goal_rows is None or not goal_rows.height:
        raise KeyError(f"No event with id {goal_event_id!r}")
    goal = goal_rows.row(0, named=True)
    chain = _possession_chain(frame, int(goal["match_id"]), int(goal["possession"]))
    chain_rows = chain.to_dicts()
    actors = list(dict.fromkeys(chain["player"].drop_nulls().to_list()))
    shot_players = chain.filter(pl.col("type") == "Shot")["player"].drop_nulls().to_list()
    scorer = shot_players[-1] if shot_players else goal.get("player")
    bounds = _chain_bounds(chain_rows)
    return {
        "event_id": goal_event_id,
        "goal_event": goal,
        "chain_events": chain_rows,
        "start_xy": bounds["start_xy"],
        "end_xy": bounds["end_xy"],
        "duration_seconds": bounds["duration_seconds"],
        "actors": actors,
        "xg": float(goal.get("shot_xg") or 0.0),
        # Convenience aliases kept for backwards compatibility.
        "match_id": int(goal["match_id"]),
        "possession": int(goal["possession"]),
        "minute": goal.get("minute"),
        "scorer": scorer,
        "n_events": int(chain.height),
        "chain": chain_rows,
    }


def assist_breakdown(frame: pl.DataFrame, assist_event_id: str) -> dict[str, Any]:
    """Reconstruct the possession chain leading to an assist."""

    assist_rows = _find_event(frame, assist_event_id)
    if assist_rows is None or not assist_rows.height:
        raise KeyError(f"No event with id {assist_event_id!r}")
    assist = assist_rows.row(0, named=True)
    chain = _possession_chain(frame, int(assist["match_id"]), int(assist["possession"]))
    return {
        "event_id": assist_event_id,
        "match_id": int(assist["match_id"]),
        "possession": int(assist["possession"]),
        "minute": assist.get("minute"),
        "assister": assist.get("player"),
        "recipient": assist.get("pass_recipient"),
        "n_events": int(chain.height),
        "chain": chain.to_dicts(),
    }


def rolling_form(
    frame: pl.DataFrame, *, window: int = 5, player_id: int | None = MESSI_ID
) -> pl.DataFrame:
    """Return goals and assists per match with a rolling ``window`` series."""

    mine = _player_filter(frame, player_id)
    if not mine.height:
        return pl.DataFrame(
            schema={
                "match_id": pl.Int64,
                "match_date": pl.Utf8,
                "goals": pl.Int64,
                "assists": pl.Int64,
                "goal_contributions": pl.Int64,
            }
        )
    per_match = (
        mine.group_by("match_id")
        .agg(
            [
                pl.col("match_date").first().alias("match_date"),
                pl.col("is_goal").fill_null(False).sum().alias("goals"),
                pl.col("is_assist").fill_null(False).sum().alias("assists"),
                pl.col("season_name").first().alias("season_name"),
            ]
        )
        .sort("match_date")
        .with_columns((pl.col("goals") + pl.col("assists")).alias("goal_contributions"))
    )
    return per_match.with_columns(
        [
            pl.col("goal_contributions")
            .rolling_mean(window_size=window, min_samples=1)
            .alias(f"rolling_{window}_contributions")
        ]
    )
