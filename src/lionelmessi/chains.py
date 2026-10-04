"""Possession-chain ("continuation") analysis.

A *continuation* is an ordered sequence of events belonging to the same
possession, broken whenever the possession changes, a dead-ball restart occurs,
or a large time gap elapses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from lionelmessi.types import Continuation, Event

if TYPE_CHECKING:  # pragma: no cover
    from lionelmessi.models import ExpectedThreat

__all__ = [
    "build_continuations",
    "filter_continuations",
    "continuation_summary",
    "DEAD_BALL_TYPES",
]

#: Pass types that always begin a fresh continuation.
DEAD_BALL_TYPES: frozenset[str] = frozenset(
    {
        "Throw-in",
        "Corner",
        "Free Kick",
        "Goal Kick",
        "Kick Off",
        "Penalty",
    }
)

#: Event types that terminate the current continuation.
_TURNOVER_TYPES: frozenset[str] = frozenset({"Interception", "Ball Recovery", "Clearance"})


#: The subset of raw fields retained on each :class:`Event` for downstream use.
_RAW_KEYS: tuple[str, ...] = (
    "location_x",
    "location_y",
    "pass_end_x",
    "pass_end_y",
    "carry_end_x",
    "carry_end_y",
    "shot_xg",
    "is_goal",
    "is_progressive",
)


def _to_event(row: dict[str, Any], match_id: int) -> Event:
    loc_x = row.get("location_x")
    loc_y = row.get("location_y")
    location: tuple[float, float] | None = (
        (float(loc_x), float(loc_y)) if loc_x is not None and loc_y is not None else None
    )
    raw = {key: row.get(key) for key in _RAW_KEYS}
    return Event(
        id=str(row.get("id")),
        type=str(row.get("type") or "Unknown"),
        index=int(row.get("index") or 0),
        period=int(row.get("period") or 1),
        timestamp=str(row.get("timestamp") or "00:00:00.000"),
        minute=int(row.get("minute") or 0),
        second=int(row.get("second") or 0),
        possession=row.get("possession"),
        possession_team=row.get("possession_team"),
        play_pattern=row.get("play_pattern"),
        team=row.get("team"),
        player=row.get("player"),
        player_id=row.get("player_id"),
        location=location,
        duration=float(row.get("duration") or 0.0),
        under_pressure=bool(row.get("under_pressure")),
        match_id=match_id,
        raw=raw,
    )


def _chain_stats(
    chain_events: list[Event],
    xt_model: ExpectedThreat | None,
) -> Continuation:
    first = chain_events[0]
    last = chain_events[-1]
    xg = 0.0
    progressive_passes = 0
    carries = 0
    shots = 0
    goal = False
    actors: list[str] = []
    for event in chain_events:
        if event.player and event.player not in actors:
            actors.append(event.player)
        raw = event.raw
        if event.type == "Shot":
            shots += 1
            xg += float(raw.get("shot_xg") or 0.0)
            goal = goal or bool(raw.get("is_goal"))
        if event.type == "Pass" and bool(raw.get("is_progressive")):
            progressive_passes += 1
        if event.type == "Carry":
            carries += 1

    xt_gain = 0.0
    if xt_model is not None and first.location and last.location:
        end_xy: tuple[float, float] | None = last.location
        raw_last = last.raw
        if raw_last.get("pass_end_x") is not None and raw_last.get("pass_end_y") is not None:
            end_xy = (float(raw_last["pass_end_x"]), float(raw_last["pass_end_y"]))
        elif raw_last.get("carry_end_x") is not None and raw_last.get("carry_end_y") is not None:
            end_xy = (float(raw_last["carry_end_x"]), float(raw_last["carry_end_y"]))
        if end_xy is not None:
            xt_gain = xt_model.xt(first.location, end_xy)

    duration = max(0.0, last.seconds - first.seconds)
    return Continuation(
        possession=int(first.possession or 0),
        match_id=first.match_id,
        events=chain_events,
        start_xy=first.location,
        end_xy=last.location,
        xg=round(xg, 4),
        xt_gain=round(xt_gain, 4),
        progressive_pass_count=progressive_passes,
        carry_count=carries,
        shot_count=shots,
        goal=goal,
        actors=actors,
        duration_seconds=duration,
    )


def build_continuations(
    events: pl.DataFrame,
    *,
    max_gap: float = 10.0,
    xt_model: ExpectedThreat | None = None,
) -> list[Continuation]:
    """Group events into ordered possession chains.

    A new chain starts when the possession changes, a dead-ball restart is
    observed, or more than ``max_gap`` seconds elapse between consecutive
    events.
    """

    if not events.height:
        return []
    frame = events.sort(["match_id", "period", "index"])
    chains: list[Continuation] = []
    current: list[Event] = []
    current_possession: int | None = None
    current_match: int | None = None
    previous_seconds: float | None = None

    def flush() -> None:
        if current:
            chains.append(_chain_stats(current, xt_model))

    for row in frame.iter_rows(named=True):
        match_id = int(row.get("match_id") or 0)
        possession = int(row.get("possession") or 0)
        event = _to_event(row, match_id)
        is_dead_ball = event.type == "Pass" and row.get("pass_type") in DEAD_BALL_TYPES
        gap_break = previous_seconds is not None and (event.seconds - previous_seconds) > max_gap
        possession_break = current and (
            possession != current_possession or match_id != current_match
        )
        if current and (possession_break or gap_break or is_dead_ball):
            flush()
            current = []
        if not current:
            current_possession = possession
            current_match = match_id
        current.append(event)
        previous_seconds = event.seconds
    flush()
    return chains


def filter_continuations(
    chains: list[Continuation],
    *,
    actor: str | None = None,
    min_xt_gain: float | None = None,
    min_xg: float | None = None,
    ended_in_goal: bool | None = None,
    min_events: int | None = None,
) -> list[Continuation]:
    """Filter chains by actor involvement, threat gained, and outcome."""

    result = chains
    if actor is not None:
        result = [c for c in result if any(actor in a for a in c.actors)]
    if min_xt_gain is not None:
        result = [c for c in result if c.xt_gain >= min_xt_gain]
    if min_xg is not None:
        result = [c for c in result if c.xg >= min_xg]
    if ended_in_goal is not None:
        result = [c for c in result if c.goal is ended_in_goal]
    if min_events is not None:
        result = [c for c in result if c.n_events >= min_events]
    return result


def continuation_summary(chains: list[Continuation]) -> pl.DataFrame:
    """Return a tabular summary of a list of continuations."""

    if not chains:
        return pl.DataFrame(
            schema={
                "match_id": pl.Int64,
                "possession": pl.Int64,
                "n_events": pl.Int64,
                "xg": pl.Float64,
                "xt_gain": pl.Float64,
                "progressive_passes": pl.Int64,
                "carries": pl.Int64,
                "shots": pl.Int64,
                "goal": pl.Boolean,
                "duration_seconds": pl.Float64,
            }
        )
    return pl.DataFrame(
        {
            "match_id": [c.match_id for c in chains],
            "possession": [c.possession for c in chains],
            "n_events": [c.n_events for c in chains],
            "xg": [c.xg for c in chains],
            "xt_gain": [c.xt_gain for c in chains],
            "progressive_passes": [c.progressive_pass_count for c in chains],
            "carries": [c.carry_count for c in chains],
            "shots": [c.shot_count for c in chains],
            "goal": [c.goal for c in chains],
            "duration_seconds": [c.duration_seconds for c in chains],
        }
    ).sort("xt_gain", descending=True)
