"""Typed domain models for :mod:`lionelmessi`.

These Pydantic models describe the structured objects that flow through the
public API.  Bulk event data stays in columnar form (Polars) for performance;
the models here are used at the boundaries where a rich, validated object is
useful (matches, single events, possession chains, action values).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Match",
    "Event",
    "Shot",
    "Pass",
    "Carry",
    "Continuation",
    "ActionValue",
]


_XY = tuple[float, float]


class Match(BaseModel):
    """A single football match."""

    model_config = ConfigDict(extra="ignore")

    match_id: int
    match_date: date | None = None
    kick_off: str | None = None
    competition_id: int | None = None
    competition_name: str | None = None
    season_id: int | None = None
    season_name: str | None = None
    home_team_id: int | None = None
    home_team_name: str | None = None
    away_team_id: int | None = None
    away_team_name: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    match_week: int | None = None


@dataclass(slots=True)
class Event:
    """A single StatsBomb event, projected onto the fields we care about.

    Implemented as a slotted dataclass (rather than a Pydantic model) because a
    full career comprises millions of events and the lighter representation
    keeps possession-chain construction practical.
    """

    id: str
    type: str
    index: int = 0
    period: int = 1
    timestamp: str = "00:00:00.000"
    minute: int = 0
    second: int = 0
    possession: int | None = None
    possession_team: str | None = None
    play_pattern: str | None = None
    team: str | None = None
    player: str | None = None
    player_id: int | None = None
    location: _XY | None = None
    duration: float = 0.0
    under_pressure: bool = False
    match_id: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        """Return the absolute event time in seconds within the match."""

        return float(self.minute * 60 + self.second)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict view of the event."""

        return asdict(self)


class Shot(BaseModel):
    """A shot event."""

    model_config = ConfigDict(extra="ignore")

    event_id: str
    match_id: int | None = None
    player: str | None = None
    location: _XY | None = None
    end_location: tuple[float, float, float] | None = None
    xg: float = 0.0
    outcome: str | None = None
    body_part: str | None = None
    technique: str | None = None

    @property
    def is_goal(self) -> bool:
        return self.outcome == "Goal"


class Pass(BaseModel):
    """A pass event."""

    model_config = ConfigDict(extra="ignore")

    event_id: str
    match_id: int | None = None
    player: str | None = None
    recipient: str | None = None
    location: _XY | None = None
    end_location: _XY | None = None
    length: float = 0.0
    angle: float = 0.0
    height: str | None = None
    body_part: str | None = None
    outcome: str | None = None
    goal_assist: bool = False
    shot_assist: bool = False

    @property
    def completed(self) -> bool:
        return self.outcome is None


class Carry(BaseModel):
    """A carry (ball progression on the ball) event."""

    model_config = ConfigDict(extra="ignore")

    event_id: str
    match_id: int | None = None
    player: str | None = None
    location: _XY | None = None
    end_location: _XY | None = None
    distance: float = 0.0
    under_pressure: bool = False


class ActionValue(BaseModel):
    """An action-value estimate for a single action."""

    model_config = ConfigDict(extra="ignore")

    event_id: str
    match_id: int | None = None
    player: str | None = None
    kind: str
    xt_start: float = 0.0
    xt_end: float = 0.0
    vaep: float = 0.0

    @property
    def xt_delta(self) -> float:
        """Expected-threat gained by the action."""

        return self.xt_end - self.xt_start


@dataclass(slots=True)
class Continuation:
    """An ordered possession chain."""

    possession: int
    match_id: int | None = None
    events: list[Event] = field(default_factory=list)
    start_xy: _XY | None = None
    end_xy: _XY | None = None
    xg: float = 0.0
    xt_gain: float = 0.0
    progressive_pass_count: int = 0
    carry_count: int = 0
    shot_count: int = 0
    goal: bool = False
    actors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def n_events(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict view of the continuation."""

        return asdict(self)
