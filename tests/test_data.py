"""Tests for the data ingestion layer."""

from __future__ import annotations

import polars as pl
import pytest

from lionelmessi import config
from lionelmessi import data as data_mod
from tests import sample_data


def test_parse_event_flattens_nested_fields() -> None:
    row = data_mod._parse_event(sample_data.RAW_EVENTS[0], sample_data.MATCH_ID)
    assert row["type"] == "Pass"
    assert row["player"] == "Lionel Andrés Messi Cuccittini"
    assert row["player_id"] == 5503
    assert row["pass_end_x"] == 110.0
    assert row["pass_goal_assist"] is True
    assert row["location_x"] == 100.0


def test_parse_event_handles_missing_optional_blocks() -> None:
    minimal = {
        "id": "x",
        "index": 1,
        "period": 1,
        "timestamp": "00:00:00.000",
        "minute": 0,
        "second": 0,
        "type": {"id": 1, "name": "Half Start"},
        "possession": 1,
    }
    row = data_mod._parse_event(minimal, 1)
    assert row["type"] == "Half Start"
    assert row["player"] is None
    assert row["pass_length"] is None
    assert row["shot_xg"] is None


def test_schema_is_stable(sample_events: pl.DataFrame) -> None:
    expected = set(data_mod._events_schema()) | {
        "is_shot",
        "is_pass",
        "is_carry",
        "is_goal",
        "is_goal_assist",
        "is_key_pass",
        "is_shot_assist",
        "is_assist",
        "is_progressive",
        "goal_number",
        "season_name",
        "competition_name",
        "match_date",
    }
    assert expected.issubset(set(sample_events.columns))


def test_derived_flags(sample_events: pl.DataFrame) -> None:
    assert int(sample_events["is_goal"].sum()) == 2
    assert int(sample_events["is_shot"].sum()) == 3
    assert int(sample_events["is_pass"].sum()) == 5
    assert int(sample_events["is_carry"].sum()) == 1
    assert int(sample_events["is_assist"].sum()) == 1
    assert int(sample_events["is_key_pass"].sum()) == 1


def test_is_messi_uses_name_candidates(sample_events: pl.DataFrame) -> None:
    assert "is_messi" in sample_events.columns
    assert int(sample_events["is_messi"].sum()) == 9
    # The name-based filter matches the id-based filter on the synthetic data.
    by_id = sample_events.filter(pl.col("player_id") == 5503)
    by_name = sample_events.filter(pl.col("is_messi"))
    assert by_id.height == by_name.height


def test_context_join_adds_season(sample_events: pl.DataFrame) -> None:
    assert sample_events["season_name"].unique().to_list() == ["2011/2012"]
    assert sample_events["competition_name"].unique().to_list() == ["La Liga"]


def test_attach_match_context_without_matches_adds_nulls(sample_events: pl.DataFrame) -> None:
    bare = sample_events.drop(["season_name", "competition_name", "match_date"])
    result = data_mod.attach_match_context(bare, matches=pl.DataFrame(), fetch=False)
    assert result["season_name"].null_count() == result.height


def test_parse_match_filters_non_messi_teams() -> None:
    other = {
        "match_id": 1,
        "home_team": {"home_team_id": 10, "home_team_name": "A"},
        "away_team": {"away_team_id": 11, "away_team_name": "B"},
        "competition": {"competition_id": 2, "competition_name": "PL"},
        "season": {"season_id": 1, "season_name": "2020/2021"},
    }
    assert data_mod._parse_match(other) is None
    barca = {
        "match_id": 2,
        "home_team": {"home_team_id": 217, "home_team_name": "Barcelona"},
        "away_team": {"away_team_id": 11, "away_team_name": "B"},
        "competition": {"competition_id": 11, "competition_name": "La Liga"},
        "season": {"season_id": 23, "season_name": "2011/2012"},
    }
    parsed = data_mod._parse_match(barca)
    assert parsed is not None
    assert parsed["season_name"] == "2011/2012"


def test_match_event_stream_groups_by_possession(sample_events: pl.DataFrame) -> None:
    stream = data_mod.MatchEventStream(sample_events, sample_data.MATCH_ID)
    assert len(stream) == sample_events.height
    grouped = stream.possessions()
    assert set(grouped) == {1, 2, 3, 4}
    assert grouped[1].height == 2
    assert stream.player("Lionel Andrés Messi Cuccittini").height == 9


def test_fetch_match_events_cache_roundtrip(
    isolated_cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_get_json(url: str, *, timeout: int | None = None) -> object:
        calls["n"] += 1
        return sample_data.RAW_EVENTS

    monkeypatch.setattr(data_mod, "_get_json", fake_get_json)

    first = data_mod.fetch_match_events(sample_data.MATCH_ID, force=True)
    assert calls["n"] == 1
    cache_file = config.cache_dir() / "events" / f"{sample_data.MATCH_ID}.parquet"
    assert cache_file.exists()

    second = data_mod.fetch_match_events(sample_data.MATCH_ID)
    assert calls["n"] == 1
    assert second.height == first.height
    assert second["is_goal"].sum() == first["is_goal"].sum()


def test_fetch_all_messi_events_empty() -> None:
    frame = data_mod.fetch_all_messi_events([], workers=1)
    assert frame.height == 0
    assert "is_goal" in frame.columns


def test_low_level_helpers() -> None:
    assert data_mod._loc([1.0, 2.0], 0) == 1.0
    assert data_mod._loc(None, 0) is None
    assert data_mod._loc(["a", "b"], 0) is None
    assert data_mod._loc([1.0], 5) is None
    assert data_mod._nested({"a": 1}, "a") == 1
    assert data_mod._nested(None, "a") is None
    assert data_mod._clean({"x": 1}) == {"x": 1}
    assert data_mod._build_session() is not None


def test_fetch_competitions_caches(isolated_cache, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [{"competition_id": 11, "season_id": 23, "competition_name": "La Liga"}]
    monkeypatch.setattr(data_mod, "_get_json", lambda url, *, timeout=None: payload)
    first = data_mod.fetch_competitions(force=True)
    assert first[0]["competition_id"] == 11

    def explode(url: str, *, timeout: int | None = None) -> object:  # pragma: no cover
        raise AssertionError("cache should have been used")

    monkeypatch.setattr(data_mod, "_get_json", explode)
    assert data_mod.fetch_competitions() == first


def test_fetch_messi_matches_filters_and_caches(
    isolated_cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    comps = [
        {"competition_id": 11, "season_id": 23},
        {"competition_id": 16, "season_id": 1},
    ]
    barca_match = {
        "match_id": 900001,
        "match_date": "2012-01-08",
        "home_team": {"home_team_id": 217, "home_team_name": "Barcelona"},
        "away_team": {"away_team_id": 207, "away_team_name": "Valencia"},
        "competition": {"competition_id": 11, "competition_name": "La Liga"},
        "season": {"season_id": 23, "season_name": "2011/2012"},
        "home_score": 2,
        "away_score": 1,
    }
    other_match = {
        "match_id": 900002,
        "match_date": "2012-02-01",
        "home_team": {"home_team_id": 10, "home_team_name": "A"},
        "away_team": {"away_team_id": 11, "away_team_name": "B"},
        "competition": {"competition_id": 11, "competition_name": "La Liga"},
        "season": {"season_id": 23, "season_name": "2011/2012"},
    }

    def fake(url: str, *, timeout: int | None = None) -> object:
        if url.endswith("competitions.json"):
            return comps
        if url.endswith("/matches/11/23.json"):
            return [barca_match, other_match]
        return []

    monkeypatch.setattr(data_mod, "_get_json", fake)
    matches = data_mod.fetch_messi_matches(force=True, verify_lineups=False)
    assert matches.height == 1
    assert matches["match_id"].to_list() == [900001]

    def explode(url: str, *, timeout: int | None = None) -> object:  # pragma: no cover
        raise AssertionError("cache should have been used")

    monkeypatch.setattr(data_mod, "_get_json", explode)
    assert data_mod.fetch_messi_matches().height == 1


def test_verify_lineups_keeps_only_messi(isolated_cache, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(url: str, *, timeout: int | None = None) -> object:
        if url.endswith("competitions.json"):
            return [{"competition_id": 11, "season_id": 23}]
        if "/matches/" in url:
            return [
                {
                    "match_id": 1,
                    "home_team": {"home_team_id": 217, "home_team_name": "Barcelona"},
                    "away_team": {"away_team_id": 207, "away_team_name": "Valencia"},
                    "competition": {"competition_id": 11, "competition_name": "La Liga"},
                    "season": {"season_id": 23, "season_name": "2011/2012"},
                },
                {
                    "match_id": 2,
                    "home_team": {"home_team_id": 217, "home_team_name": "Barcelona"},
                    "away_team": {"away_team_id": 208, "away_team_name": "Sevilla"},
                    "competition": {"competition_id": 11, "competition_name": "La Liga"},
                    "season": {"season_id": 23, "season_name": "2011/2012"},
                },
            ]
        # lineups: only match 1 features Messi
        if url.endswith("/1.json"):
            return [{"team_id": 217, "lineup": [{"player_id": 5503, "player_name": "Messi"}]}]
        return [{"team_id": 217, "lineup": [{"player_id": 99999, "player_name": "Other"}]}]

    monkeypatch.setattr(data_mod, "_get_json", fake)
    matches = data_mod.fetch_messi_matches(force=True, verify_lineups=True)
    assert matches["match_id"].to_list() == [1]


def test_fetch_all_messi_events_parallel(
    isolated_cache, monkeypatch: pytest.MonkeyPatch, sample_events: pl.DataFrame
) -> None:
    monkeypatch.setattr(data_mod, "fetch_messi_matches", lambda force=False: sample_data.MATCHES)
    monkeypatch.setattr(data_mod, "fetch_match_events", lambda mid, force=False: sample_events)
    frame = data_mod.fetch_all_messi_events(workers=3)
    assert frame.height == sample_events.height
    assert frame["season_name"].drop_nulls().unique().to_list() == ["2011/2012"]


def test_fetch_all_messi_events_skips_failures(
    isolated_cache, monkeypatch: pytest.MonkeyPatch, sample_events: pl.DataFrame
) -> None:
    import requests

    monkeypatch.setattr(data_mod, "fetch_messi_matches", lambda force=False: sample_data.MATCHES)

    def flaky(mid: int, force: bool = False):
        if mid == 2:
            raise requests.RequestException("boom")
        return sample_events

    monkeypatch.setattr(data_mod, "fetch_match_events", flaky)
    frame = data_mod.fetch_all_messi_events([1, 2], workers=2)
    assert frame.height == sample_events.height


def test_load_events_rebuild_and_read(
    isolated_cache, monkeypatch: pytest.MonkeyPatch, sample_events: pl.DataFrame
) -> None:
    monkeypatch.setattr(data_mod, "fetch_all_messi_events", lambda **k: sample_events)
    written = data_mod.load_events(rebuild=True)
    assert written.height == sample_events.height
    assert (config.cache_dir() / "events_all.parquet").exists()
    cached = data_mod.load_events()
    assert cached.height == written.height


def test_match_event_stream_iter_and_match_id(sample_events: pl.DataFrame) -> None:
    stream = data_mod.MatchEventStream(sample_events)
    assert stream.match_id == sample_data.MATCH_ID
    items = list(stream)
    assert len(items) == 4
    assert isinstance(items[0][0], int)
