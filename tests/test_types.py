"""Tests for the typed domain models."""

from __future__ import annotations

import json

import pytest

from lionelmessi.types import ActionValue, Carry, Continuation, Event, Match, Pass, Shot


def test_match_roundtrip() -> None:
    match = Match(match_id=1, competition_name="La Liga", season_name="2011/2012")
    dumped = match.model_dump()
    assert dumped["match_id"] == 1
    assert Match(**dumped) == match
    assert json.loads(match.model_dump_json())["match_id"] == 1


def test_shot_pass_carry_models() -> None:
    shot = Shot(event_id="s1", outcome="Goal", xg=0.4)
    assert shot.is_goal
    assert Shot(event_id="s2", outcome="Saved").is_goal is False
    assert json.loads(shot.model_dump_json())["xg"] == 0.4

    completed = Pass(event_id="p1")
    incomplete = Pass(event_id="p2", outcome="Incomplete")
    assert completed.completed
    assert not incomplete.completed

    carry = Carry(event_id="c1", distance=12.5)
    assert carry.model_dump()["distance"] == 12.5


def test_action_value_delta() -> None:
    value = ActionValue(event_id="a1", kind="move", xt_start=0.05, xt_end=0.12)
    assert value.xt_delta == pytest.approx(0.07)
    assert json.loads(value.model_dump_json())["kind"] == "move"


def test_event_and_continuation_serialize() -> None:
    event = Event(id="e1", type="Pass", minute=2, second=30, location=(60.0, 40.0))
    assert event.seconds == 150.0
    event_dict = event.to_dict()
    assert event_dict["id"] == "e1"
    assert event_dict["location"] == (60.0, 40.0)
    assert json.dumps(event_dict)

    chain = Continuation(possession=3, events=[event], actors=["Messi"], goal=True)
    assert chain.n_events == 1
    chain_dict = chain.to_dict()
    assert chain_dict["possession"] == 3
    assert chain_dict["events"][0]["id"] == "e1"
    assert json.dumps(chain_dict)
