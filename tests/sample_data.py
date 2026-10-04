"""Deterministic synthetic StatsBomb-like events for offline tests.

These mimic the raw JSON shape from StatsBomb Open Data closely enough to
exercise the real parser and the whole downstream pipeline without a network
connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from lionelmessi import data

MATCH_ID = 900001

_MESSI = {"id": 5503, "name": "Lionel Andrés Messi Cuccittini"}
_BUSQUETS = {"id": 5203, "name": "Sergio Busquets i Burgos"}
_ALBA = {"id": 5211, "name": "Jordi Alba Ramos"}
_BARCA = {"id": 217, "name": "Barcelona"}
_OPP = {"id": 207, "name": "Valencia"}


def _event(
    index: int,
    minute: int,
    second: int,
    type_name: str,
    possession: int,
    player: dict[str, Any],
    location: list[float],
    pass_: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": f"evt-{index:03d}",
        "index": index,
        "period": 1,
        "timestamp": f"00:{minute:02d}:{second:02d}.000",
        "minute": minute,
        "second": second,
        "type": {"id": 1, "name": type_name},
        "possession": possession,
        "possession_team": _BARCA,
        "play_pattern": {"id": 1, "name": "Regular Play"},
        "team": _BARCA if player is not None else _OPP,
        "player": player,
        "location": location,
        "duration": 0.5,
    }
    row.update(extra)
    if pass_ is not None:
        row["pass"] = pass_
    return row


RAW_EVENTS: list[dict[str, Any]] = [
    # Possession 1: Messi assist -> Busquets goal.
    _event(
        1,
        0,
        5,
        "Pass",
        1,
        _MESSI,
        [100.0, 40.0],
        pass_={
            "recipient": _BUSQUETS,
            "length": 10.0,
            "angle": 0.0,
            "height": {"id": 1, "name": "Ground Pass"},
            "end_location": [110.0, 40.0],
            "body_part": {"id": 40, "name": "Right Foot"},
            "goal_assist": True,
            "shot_assist": True,
        },
    ),
    _event(
        2,
        0,
        8,
        "Shot",
        1,
        _BUSQUETS,
        [110.0, 40.0],
        shot={
            "statsbomb_xg": 0.4,
            "end_location": [120.0, 40.0],
            "outcome": {"id": 97, "name": "Goal"},
            "body_part": {"id": 40, "name": "Right Foot"},
            "technique": {"id": 93, "name": "Normal"},
        },
    ),
    # Possession 2: Messi progressive pass + carry + goal.
    _event(
        3,
        1,
        0,
        "Pass",
        2,
        _MESSI,
        [60.0, 20.0],
        pass_={
            "recipient": _ALBA,
            "length": 15.0,
            "angle": 0.32,
            "height": {"id": 1, "name": "Ground Pass"},
            "end_location": [75.0, 25.0],
            "body_part": {"id": 38, "name": "Left Foot"},
        },
    ),
    _event(4, 1, 3, "Carry", 2, _MESSI, [75.0, 25.0], carry={"end_location": [88.0, 30.0]}),
    _event(
        5,
        1,
        6,
        "Shot",
        2,
        _MESSI,
        [88.0, 30.0],
        shot={
            "statsbomb_xg": 0.2,
            "end_location": [120.0, 38.0],
            "outcome": {"id": 97, "name": "Goal"},
            "body_part": {"id": 38, "name": "Left Foot"},
            "technique": {"id": 93, "name": "Normal"},
        },
    ),
    # Possession 3: backward pass, dribbles, teammate action.
    _event(
        6,
        2,
        0,
        "Pass",
        3,
        _MESSI,
        [40.0, 40.0],
        pass_={
            "recipient": _BUSQUETS,
            "length": 10.0,
            "angle": 3.14,
            "height": {"id": 1, "name": "Ground Pass"},
            "end_location": [30.0, 40.0],
            "body_part": {"id": 38, "name": "Left Foot"},
        },
    ),
    _event(
        7,
        2,
        3,
        "Dribble",
        3,
        _MESSI,
        [30.0, 40.0],
        dribble={"outcome": {"id": 8, "name": "Complete"}},
    ),
    _event(
        8,
        2,
        5,
        "Dribble",
        3,
        _MESSI,
        [35.0, 45.0],
        dribble={"outcome": {"id": 9, "name": "Incomplete"}},
    ),
    _event(
        9,
        2,
        7,
        "Pass",
        3,
        _ALBA,
        [30.0, 40.0],
        pass_={
            "recipient": _MESSI,
            "length": 22.4,
            "angle": 0.46,
            "height": {"id": 1, "name": "Ground Pass"},
            "end_location": [50.0, 50.0],
            "body_part": {"id": 38, "name": "Left Foot"},
        },
    ),
    # Possession 4: dead-ball corner -> saved shot.
    _event(
        10,
        3,
        0,
        "Pass",
        4,
        _MESSI,
        [80.0, 50.0],
        pass_={
            "recipient": _BUSQUETS,
            "length": 21.0,
            "angle": -0.24,
            "height": {"id": 3, "name": "High Pass"},
            "type": {"id": 61, "name": "Corner"},
            "end_location": [100.0, 45.0],
            "body_part": {"id": 38, "name": "Left Foot"},
        },
    ),
    _event(
        11,
        3,
        4,
        "Shot",
        4,
        _MESSI,
        [100.0, 45.0],
        shot={
            "statsbomb_xg": 0.1,
            "end_location": [119.0, 44.0],
            "outcome": {"id": 100, "name": "Saved"},
            "body_part": {"id": 38, "name": "Left Foot"},
            "technique": {"id": 93, "name": "Normal"},
        },
    ),
]

MATCHES: pl.DataFrame = pl.DataFrame(
    [
        {
            "match_id": MATCH_ID,
            "match_date": "2012-01-08",
            "kick_off": "20:00:00.000",
            "competition_id": 11,
            "competition_name": "La Liga",
            "season_id": 23,
            "season_name": "2011/2012",
            "home_team_id": 217,
            "home_team_name": "Barcelona",
            "away_team_id": 207,
            "away_team_name": "Valencia",
            "home_score": 2,
            "away_score": 1,
            "match_week": 18,
        }
    ],
    schema={
        "match_id": pl.Int64,
        "match_date": pl.Utf8,
        "kick_off": pl.Utf8,
        "competition_id": pl.Int64,
        "competition_name": pl.Utf8,
        "season_id": pl.Int64,
        "season_name": pl.Utf8,
        "home_team_id": pl.Int64,
        "home_team_name": pl.Utf8,
        "away_team_id": pl.Int64,
        "away_team_name": pl.Utf8,
        "home_score": pl.Int64,
        "away_score": pl.Int64,
        "match_week": pl.Int64,
    },
)


def build_sample() -> pl.DataFrame:
    """Parse the synthetic raw events into the canonical event frame."""

    rows = [data._parse_event(raw, MATCH_ID) for raw in RAW_EVENTS]  # noqa: SLF001
    frame = pl.DataFrame(rows, schema=data._events_schema())  # noqa: SLF001
    frame = frame.sort(["period", "index"])
    frame = data.add_derived_columns(frame)
    return data.attach_match_context(frame, matches=MATCHES)


def fixture_path() -> Path:
    """Return the on-disk Parquet fixture path."""

    return Path(__file__).parent / "data" / "sample_events.parquet"


def load_or_build() -> pl.DataFrame:
    """Load the Parquet fixture, building and writing it if absent."""

    path = fixture_path()
    if path.exists():
        return pl.read_parquet(path)
    frame = build_sample()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame
