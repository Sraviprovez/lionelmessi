"""Tests for the configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from lionelmessi import config


def test_cache_dir_override_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "cache"
    result = config.cache_dir(override=target)
    assert result == target
    assert result.is_dir()


def test_env_int_parses_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM10_TEST_INT", "7")
    assert config._env_int("LM10_TEST_INT", 1) == 7
    monkeypatch.setenv("LM10_TEST_INT", "not-a-number")
    assert config._env_int("LM10_TEST_INT", 1) == 1
    monkeypatch.delenv("LM10_TEST_INT")
    assert config._env_int("LM10_TEST_INT", 3) == 3


def test_env_float_parses_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM10_TEST_FLOAT", "0.25")
    assert config._env_float("LM10_TEST_FLOAT", 1.0) == pytest.approx(0.25)
    monkeypatch.setenv("LM10_TEST_FLOAT", "x")
    assert config._env_float("LM10_TEST_FLOAT", 1.0) == 1.0


def test_env_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LM10_TEST_PATH", str(tmp_path))
    assert config._env_path("LM10_TEST_PATH", Path("/default")) == tmp_path
    monkeypatch.delenv("LM10_TEST_PATH")
    assert config._env_path("LM10_TEST_PATH", Path("/default")) == Path("/default")


def test_static_constants_are_consistent() -> None:
    assert config.PITCH_HALF_LENGTH == config.PITCH_LENGTH / 2
    assert config.MESSI_PLAYER_ID == 5503
    assert 217 in config.MESSI_TEAM_IDS
    assert isinstance(config.SB_SOURCE_NOTE, str)
