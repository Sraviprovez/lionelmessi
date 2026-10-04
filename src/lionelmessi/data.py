"""StatsBomb Open Data ingestion, parsing, and on-disk caching.

The public entry points fetch Messi's matches and events, flatten the nested
StatsBomb JSON into a tidy columnar frame, derive convenience flags, and cache
everything as Parquet so that repeated runs are deterministic and offline.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lionelmessi import config

logger = logging.getLogger(__name__)

Json = Any

__all__ = [
    "fetch_competitions",
    "fetch_messi_matches",
    "fetch_match_events",
    "fetch_all_messi_events",
    "load_events",
    "add_derived_columns",
    "attach_match_context",
    "MatchEventStream",
    "EVENT_COLUMNS",
]

#: Target point used to measure progression towards goal (centre of the goal).
GOAL_CENTER: tuple[float, float] = (config.PITCH_LENGTH, config.PITCH_WIDTH / 2)

#: Canonical column order for the flattened event frame.
EVENT_COLUMNS: list[str] = [
    "id",
    "match_id",
    "index",
    "period",
    "timestamp",
    "minute",
    "second",
    "type",
    "possession",
    "possession_team",
    "possession_team_id",
    "play_pattern",
    "team",
    "team_id",
    "player",
    "player_id",
    "location_x",
    "location_y",
    "duration",
    "under_pressure",
    "pass_recipient",
    "pass_length",
    "pass_angle",
    "pass_height",
    "pass_end_x",
    "pass_end_y",
    "pass_body_part",
    "pass_outcome",
    "pass_type",
    "pass_shot_assist",
    "pass_goal_assist",
    "shot_xg",
    "shot_end_x",
    "shot_end_y",
    "shot_outcome",
    "shot_body_part",
    "shot_technique",
    "shot_type",
    "carry_end_x",
    "carry_end_y",
    "dribble_outcome",
    "goalkeeper_type",
    "goalkeeper_outcome",
    "ball_receipt_outcome",
    "ball_recovery_failure",
    "duel_type",
    "duel_outcome",
    "interception_outcome",
    "clearance_body_part",
    "foul_committed_card",
    "foul_won_defensive",
]


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------
def _build_session() -> requests.Session:
    """Return a :class:`requests.Session` with retries and backoff."""

    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=config.MAX_WORKERS * 2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "lionelmessi/0.1.0 (+https://github.com/lionelmessi)"})
    return session


_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _build_session()
    return _SESSION


def _get_json(url: str, *, timeout: int | None = None) -> Json:
    """Fetch ``url`` and decode the JSON body, raising for HTTP errors."""

    response = _session().get(url, timeout=timeout or config.TIMEOUT)
    response.raise_for_status()
    return cast(Json, response.json())


def _get_json_cached(url: str, path: Path, *, force: bool = False) -> Json:
    """Fetch ``url`` through a JSON file cache at ``path``."""

    if path.exists() and not force:
        return cast(Json, json.loads(path.read_text(encoding="utf-8")))
    data = _get_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _clean(value: Json) -> Json:
    """Return ``value`` unchanged; kept as a hook for future normalisation."""

    return value


# ---------------------------------------------------------------------------
# Competitions & matches
# ---------------------------------------------------------------------------
def fetch_competitions(*, force: bool = False) -> list[dict[str, Any]]:
    """Return the StatsBomb competition/season catalogue, cached on disk."""

    url = f"{config.SB_BASE}/competitions.json"
    path = config.cache_dir() / "competitions.json"
    data = _get_json_cached(url, path, force=force)
    if not isinstance(data, list):  # pragma: no cover - defensive
        raise ValueError("Unexpected competitions payload")
    return [cast(dict[str, Any], row) for row in data]


def _candidate_seasons() -> list[dict[str, Any]]:
    """Return competition/season entries worth scanning for Messi."""

    comps = fetch_competitions()
    return [c for c in comps if c.get("competition_id") in config.MESSI_COMPETITION_IDS]


def _matches_url(competition_id: int, season_id: int) -> str:
    return f"{config.SB_BASE}/matches/{competition_id}/{season_id}.json"


def _parse_match(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Project a raw match record onto the columns we keep."""

    home = _clean(raw.get("home_team")) or {}
    away = _clean(raw.get("away_team")) or {}
    competition = _clean(raw.get("competition")) or {}
    season = _clean(raw.get("season")) or {}
    home_id = home.get("home_team_id")
    away_id = away.get("away_team_id")
    if home_id not in config.MESSI_TEAM_IDS and away_id not in config.MESSI_TEAM_IDS:
        return None
    return {
        "match_id": raw.get("match_id"),
        "match_date": raw.get("match_date"),
        "kick_off": raw.get("kick_off"),
        "competition_id": competition.get("competition_id"),
        "competition_name": competition.get("competition_name"),
        "season_id": season.get("season_id"),
        "season_name": season.get("season_name"),
        "home_team_id": home_id,
        "home_team_name": home.get("home_team_name"),
        "away_team_id": away_id,
        "away_team_name": away.get("away_team_name"),
        "home_score": raw.get("home_score"),
        "away_score": raw.get("away_score"),
        "match_week": raw.get("match_week"),
    }


def _matches_frame_schema() -> dict[str, Any]:
    return {
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
    }


def fetch_messi_matches(force: bool = False, *, verify_lineups: bool = True) -> pl.DataFrame:
    """Return every Messi match available in StatsBomb Open Data.

    The result is cached at ``<cache>/matches.parquet``.  Pass ``force=True`` to
    ignore the cache and re-download.  When ``verify_lineups`` is true (the
    default) matches are kept only if Messi appears in the confirmed lineup.
    """

    cache_path = config.cache_dir() / "matches.parquet"
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)

    entries = _candidate_seasons()
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()

    for entry in entries:
        competition_id = int(entry["competition_id"])
        season_id = int(entry["season_id"])
        try:
            payload = _get_json(_matches_url(competition_id, season_id))
        except requests.RequestException as exc:  # pragma: no cover - network
            logger.warning("Skipping %s/%s: %s", competition_id, season_id, exc)
            continue
        if not isinstance(payload, list):
            continue
        for raw in payload:
            parsed = _parse_match(cast(dict[str, Any], raw))
            if parsed is None or parsed["match_id"] in seen:
                continue
            seen.add(int(parsed["match_id"]))
            rows.append(parsed)

    if verify_lineups and rows:
        rows = _verify_lineups(rows, workers=config.MAX_WORKERS)

    frame = (
        pl.DataFrame(rows, schema=_matches_frame_schema())
        if rows
        else pl.DataFrame(schema=_matches_frame_schema())
    )
    frame = frame.sort("match_date") if frame.height else frame
    frame.write_parquet(cache_path)
    return frame


def _lineup_has_messi(match_id: int) -> bool:
    """Return True if Messi appears in either team's lineup for ``match_id``."""

    path = config.cache_dir() / "lineups" / f"{match_id}.json"
    try:
        payload = _get_json_cached(f"{config.SB_BASE}/lineups/{match_id}.json", path)
    except requests.RequestException:  # pragma: no cover - network
        return True  # keep the match rather than silently dropping it
    if not isinstance(payload, list):  # pragma: no cover - defensive
        return True
    for team in payload:
        for player in team.get("lineup", []):
            if player.get("player_id") == config.MESSI_PLAYER_ID:
                return True
    return False


def _verify_lineups(
    rows: list[dict[str, Any]], *, workers: int = config.MAX_WORKERS
) -> list[dict[str, Any]]:
    """Keep only matches where Messi actually featured."""

    kept: list[dict[str, Any]] = []
    match_ids = [int(row["match_id"]) for row in rows]
    featured: dict[int, bool] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_lineup_has_messi, mid): mid for mid in match_ids}
        for start in range(0, len(match_ids), max(1, workers)):
            if start:
                time.sleep(config.BATCH_DELAY)
        for future in as_completed(futures):
            featured[futures[future]] = future.result()
    for row in rows:
        if featured.get(int(row["match_id"]), True):
            kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
def _loc(value: Json, index: int) -> float | None:
    if isinstance(value, (list, tuple)) and len(value) > index:
        try:
            return float(value[index])
        except (TypeError, ValueError):
            return None
    return None


def _nested(value: Json, key: str) -> Json:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _parse_event(raw: dict[str, Any], match_id: int) -> dict[str, Any]:
    """Flatten a single StatsBomb event into the tidy column schema."""

    event_type = _nested(raw.get("type"), "name")
    team = _clean(raw.get("team")) or {}
    player = _clean(raw.get("player")) or {}
    possession_team = _clean(raw.get("possession_team")) or {}
    play_pattern = _clean(raw.get("play_pattern")) or {}
    pass_ = _clean(raw.get("pass"))
    shot = _clean(raw.get("shot"))
    carry = _clean(raw.get("carry"))
    dribble = _clean(raw.get("dribble"))
    goalkeeper = _clean(raw.get("goalkeeper"))
    receipt = _clean(raw.get("ball_receipt"))
    recovery = _clean(raw.get("ball_recovery"))
    duel = _clean(raw.get("duel"))
    interception = _clean(raw.get("interception"))
    clearance = _clean(raw.get("clearance"))
    foul_committed = _clean(raw.get("foul_committed"))
    foul_won = _clean(raw.get("foul_won"))
    location = _clean(raw.get("location"))

    row: dict[str, Any] = {
        "id": raw.get("id"),
        "match_id": match_id,
        "index": _clean(raw.get("index")),
        "period": _clean(raw.get("period")),
        "timestamp": _clean(raw.get("timestamp")),
        "minute": _clean(raw.get("minute")),
        "second": _clean(raw.get("second")),
        "type": event_type,
        "possession": _clean(raw.get("possession")),
        "possession_team": _nested(possession_team, "name"),
        "possession_team_id": _nested(possession_team, "id"),
        "play_pattern": _nested(play_pattern, "name"),
        "team": _nested(team, "name"),
        "team_id": _nested(team, "id"),
        "player": _nested(player, "name"),
        "player_id": _nested(player, "id"),
        "location_x": _loc(location, 0),
        "location_y": _loc(location, 1),
        "duration": _clean(raw.get("duration")),
        "under_pressure": bool(raw.get("under_pressure", False)),
        "pass_recipient": _nested(_nested(pass_, "recipient"), "name") if pass_ else None,
        "pass_length": _nested(pass_, "length") if pass_ else None,
        "pass_angle": _nested(pass_, "angle") if pass_ else None,
        "pass_height": _nested(_nested(pass_, "height"), "name") if pass_ else None,
        "pass_end_x": _loc(_nested(pass_, "end_location"), 0) if pass_ else None,
        "pass_end_y": _loc(_nested(pass_, "end_location"), 1) if pass_ else None,
        "pass_body_part": _nested(_nested(pass_, "body_part"), "name") if pass_ else None,
        "pass_outcome": _nested(_nested(pass_, "outcome"), "name") if pass_ else None,
        "pass_type": _nested(_nested(pass_, "type"), "name") if pass_ else None,
        "pass_shot_assist": bool(_nested(pass_, "shot_assist")) if pass_ else False,
        "pass_goal_assist": bool(_nested(pass_, "goal_assist")) if pass_ else False,
        "shot_xg": _nested(shot, "statsbomb_xg") if shot else None,
        "shot_end_x": _loc(_nested(shot, "end_location"), 0) if shot else None,
        "shot_end_y": _loc(_nested(shot, "end_location"), 1) if shot else None,
        "shot_outcome": _nested(_nested(shot, "outcome"), "name") if shot else None,
        "shot_body_part": _nested(_nested(shot, "body_part"), "name") if shot else None,
        "shot_technique": _nested(_nested(shot, "technique"), "name") if shot else None,
        "shot_type": _nested(_nested(shot, "type"), "name") if shot else None,
        "carry_end_x": _loc(_nested(carry, "end_location"), 0) if carry else None,
        "carry_end_y": _loc(_nested(carry, "end_location"), 1) if carry else None,
        "dribble_outcome": _nested(_nested(dribble, "outcome"), "name") if dribble else None,
        "goalkeeper_type": _nested(_nested(goalkeeper, "type"), "name") if goalkeeper else None,
        "goalkeeper_outcome": _nested(_nested(goalkeeper, "outcome"), "name")
        if goalkeeper
        else None,
        "ball_receipt_outcome": _nested(_nested(receipt, "outcome"), "name") if receipt else None,
        "ball_recovery_failure": bool(_nested(recovery, "recovery_failure")) if recovery else None,
        "duel_type": _nested(_nested(duel, "type"), "name") if duel else None,
        "duel_outcome": _nested(_nested(duel, "outcome"), "name") if duel else None,
        "interception_outcome": _nested(_nested(interception, "outcome"), "name")
        if interception
        else None,
        "clearance_body_part": _nested(_nested(clearance, "body_part"), "name")
        if clearance
        else None,
        "foul_committed_card": _nested(_nested(foul_committed, "card"), "name")
        if foul_committed
        else None,
        "foul_won_defensive": bool(_nested(foul_won, "defensive")) if foul_won else None,
    }
    return row


def _events_schema() -> dict[str, Any]:
    return {
        "id": pl.Utf8,
        "match_id": pl.Int64,
        "index": pl.Int64,
        "period": pl.Int64,
        "timestamp": pl.Utf8,
        "minute": pl.Int64,
        "second": pl.Int64,
        "type": pl.Utf8,
        "possession": pl.Int64,
        "possession_team": pl.Utf8,
        "possession_team_id": pl.Int64,
        "play_pattern": pl.Utf8,
        "team": pl.Utf8,
        "team_id": pl.Int64,
        "player": pl.Utf8,
        "player_id": pl.Int64,
        "location_x": pl.Float64,
        "location_y": pl.Float64,
        "duration": pl.Float64,
        "under_pressure": pl.Boolean,
        "pass_recipient": pl.Utf8,
        "pass_length": pl.Float64,
        "pass_angle": pl.Float64,
        "pass_height": pl.Utf8,
        "pass_end_x": pl.Float64,
        "pass_end_y": pl.Float64,
        "pass_body_part": pl.Utf8,
        "pass_outcome": pl.Utf8,
        "pass_type": pl.Utf8,
        "pass_shot_assist": pl.Boolean,
        "pass_goal_assist": pl.Boolean,
        "shot_xg": pl.Float64,
        "shot_end_x": pl.Float64,
        "shot_end_y": pl.Float64,
        "shot_outcome": pl.Utf8,
        "shot_body_part": pl.Utf8,
        "shot_technique": pl.Utf8,
        "shot_type": pl.Utf8,
        "carry_end_x": pl.Float64,
        "carry_end_y": pl.Float64,
        "dribble_outcome": pl.Utf8,
        "goalkeeper_type": pl.Utf8,
        "goalkeeper_outcome": pl.Utf8,
        "ball_receipt_outcome": pl.Utf8,
        "ball_recovery_failure": pl.Boolean,
        "duel_type": pl.Utf8,
        "duel_outcome": pl.Utf8,
        "interception_outcome": pl.Utf8,
        "clearance_body_part": pl.Utf8,
        "foul_committed_card": pl.Utf8,
        "foul_won_defensive": pl.Boolean,
    }


def _distance_to_goal(x: float | None, y: float | None) -> float | None:
    if x is None or y is None:
        return None
    return math.hypot(config.PITCH_LENGTH - x, config.PITCH_WIDTH / 2 - y)


def add_derived_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the boolean convenience flags used across the package."""

    start_dist = pl.struct(["location_x", "location_y"]).map_elements(
        lambda s: _distance_to_goal(s["location_x"], s["location_y"]), return_dtype=pl.Float64
    )
    in_box = (
        (pl.col("location_x") >= 102) & (pl.col("location_y") >= 18) & (pl.col("location_y") <= 62)
    )

    end_x = pl.coalesce([pl.col("pass_end_x"), pl.col("carry_end_x")])
    end_y = pl.coalesce([pl.col("pass_end_y"), pl.col("carry_end_y")])
    end_dist = pl.struct([end_x.alias("ex"), end_y.alias("ey")]).map_elements(
        lambda s: _distance_to_goal(s["ex"], s["ey"]), return_dtype=pl.Float64
    )
    end_in_box = ((pl.col("pass_end_x") >= 102) | (pl.col("carry_end_x") >= 102)) & (
        (pl.col("pass_end_y").is_between(18, 62)) | (pl.col("carry_end_y").is_between(18, 62))
    )

    frame = frame.with_columns(
        [
            (pl.col("type") == "Shot").alias("is_shot"),
            (pl.col("type") == "Pass").alias("is_pass"),
            (pl.col("type") == "Carry").alias("is_carry"),
            ((pl.col("type") == "Shot") & (pl.col("shot_outcome") == "Goal")).alias("is_goal"),
            (
                (pl.col("player_id") == config.MESSI_PLAYER_ID)
                | pl.col("player").is_in(list(config.MESSI_NAME_CANDIDATES))
            )
            .fill_null(False)
            .alias("is_messi"),
            (
                (pl.col("type") == "Pass") & (pl.col("pass_goal_assist") == True)  # noqa: E712
            ).alias("is_goal_assist"),
            (
                (pl.col("type") == "Pass") & (pl.col("pass_shot_assist") == True)  # noqa: E712
            ).alias("is_key_pass"),
            (
                (pl.col("type") == "Pass")
                & (pl.col("pass_shot_assist") == True)  # noqa: E712
                & (pl.col("pass_goal_assist") == False)  # noqa: E712
            ).alias("is_shot_assist"),
            (
                (pl.col("type") == "Pass") & (pl.col("pass_goal_assist") == True)  # noqa: E712
            ).alias("is_assist"),
        ]
    )
    progressive = pl.col("is_shot") | (
        (pl.col("is_pass") | pl.col("is_carry"))
        & (((start_dist - end_dist) >= 10.0) | (in_box & ~end_in_box) | end_in_box)
    )
    frame = frame.with_columns(
        [
            progressive.fill_null(False).alias("is_progressive"),
            pl.col("is_goal").cum_sum().over("match_id").alias("goal_number"),
        ]
    )
    return frame


def fetch_match_events(match_id: int, force: bool = False) -> pl.DataFrame:
    """Fetch and cache the flattened event stream for one match.

    The returned frame has the derived boolean flags attached and the match
    context (season, competition, date) joined in when the match cache is
    available locally.
    """

    cache_path = config.cache_dir() / "events" / f"{match_id}.parquet"
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)

    url = f"{config.SB_BASE}/events/{match_id}.json"
    payload = _get_json(url)
    if not isinstance(payload, list):  # pragma: no cover - defensive
        raise ValueError(f"Unexpected events payload for match {match_id}")
    rows = [_parse_event(cast(dict[str, Any], raw), match_id) for raw in payload]
    frame = pl.DataFrame(rows, schema=_events_schema())
    frame = frame.sort(["period", "index"])
    frame = attach_match_context(add_derived_columns(frame), fetch=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(cache_path)
    return frame


def fetch_all_messi_events(
    match_ids: Sequence[int] | None = None,
    workers: int = config.MAX_WORKERS,
    *,
    force: bool = False,
) -> pl.DataFrame:
    """Fetch every Messi match's events in parallel, with light rate limiting."""

    if match_ids is None:
        match_ids = fetch_messi_matches(force=force)["match_id"].to_list()

    unique_ids = sorted({int(mid) for mid in match_ids})
    frames: dict[int, pl.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch_match_events, mid, force=force): mid for mid in unique_ids}
        for batch_start in range(0, len(unique_ids), max(1, workers)):
            if batch_start:
                time.sleep(config.BATCH_DELAY)
        for future in as_completed(futures):
            mid = futures[future]
            try:
                frames[mid] = future.result()
            except requests.RequestException as exc:  # pragma: no cover - network
                logger.warning("Failed to fetch events for match %s: %s", mid, exc)

    if not frames:
        return _empty_events_frame()
    combined = pl.concat([frames[mid] for mid in unique_ids if mid in frames], how="vertical")
    combined = attach_match_context(add_derived_columns(combined), fetch=True)
    return combined


def attach_match_context(
    frame: pl.DataFrame,
    *,
    matches: pl.DataFrame | None = None,
    fetch: bool = False,
) -> pl.DataFrame:
    """Join season/competition/date context onto an event frame.

    When ``fetch`` is true the match catalogue is downloaded if needed;
    otherwise only an existing on-disk catalogue is used and null columns are
    added as a fallback so the schema stays stable.
    """

    context_cols = ["season_name", "competition_name", "match_date"]
    frame = frame.drop([c for c in context_cols if c in frame.columns])

    if matches is None:
        cache_path = config.cache_dir() / "matches.parquet"
        if cache_path.exists():
            matches = pl.read_parquet(cache_path)
        elif fetch:
            matches = fetch_messi_matches()
    if matches is None or not matches.height:
        return frame.with_columns([pl.lit(None, dtype=pl.Utf8).alias(c) for c in context_cols])
    meta = matches.select(["match_id", *context_cols])
    return frame.join(meta, on="match_id", how="left")


def _empty_events_frame() -> pl.DataFrame:
    schema = _events_schema()
    for flag in (
        "is_shot",
        "is_pass",
        "is_carry",
        "is_goal",
        "is_assist",
        "is_goal_assist",
        "is_key_pass",
        "is_shot_assist",
        "is_progressive",
        "is_messi",
    ):
        schema[flag] = pl.Boolean
    schema["goal_number"] = pl.UInt32
    schema["season_name"] = pl.Utf8
    schema["competition_name"] = pl.Utf8
    schema["match_date"] = pl.Utf8
    return pl.DataFrame(schema=schema)


def load_events(rebuild: bool = False, *, workers: int = config.MAX_WORKERS) -> pl.DataFrame:
    """Load the combined Messi event cache, building it when missing.

    The combined frame is stored at ``<cache>/events_all.parquet`` so that a
    warm cache never touches the network.
    """

    combined_path = config.cache_dir() / "events_all.parquet"
    if combined_path.exists() and not rebuild:
        return pl.read_parquet(combined_path)
    frame = fetch_all_messi_events(workers=workers, force=rebuild)
    frame.write_parquet(combined_path)
    return frame


class MatchEventStream:
    """A filtered, chronologically sorted view over one match's events."""

    def __init__(self, events: pl.DataFrame, match_id: int | None = None) -> None:
        frame = events
        if match_id is not None:
            frame = frame.filter(pl.col("match_id") == match_id)
        self.events = frame.sort(["period", "index"])
        self.match_id = (
            match_id
            if match_id is not None
            else (self.events["match_id"][0] if self.events.height else None)
        )

    def player(self, name: str) -> pl.DataFrame:
        """Return only the rows attributable to ``name``."""

        return self.events.filter(pl.col("player") == name)

    def possessions(self) -> dict[int, pl.DataFrame]:
        """Group the events by possession id, preserving order."""

        grouped: dict[int, pl.DataFrame] = {}
        if not self.events.height:
            return grouped
        partitions = self.events.partition_by("possession", maintain_order=True, as_dict=True)
        for key, part in partitions.items():
            possession = int(key[0]) if isinstance(key, tuple) else int(cast(int, key))
            grouped[possession] = part
        return grouped

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.events.height

    def __iter__(self) -> Iterable[tuple[int, pl.DataFrame]]:  # pragma: no cover - trivial
        return iter(self.possessions().items())
