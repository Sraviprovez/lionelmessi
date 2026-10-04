"""CLI tests exercised through Typer's CliRunner."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from lionelmessi import data as data_mod
from lionelmessi.cli import app

runner = CliRunner()


@pytest.fixture()
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_events: pl.DataFrame) -> Path:
    """Route the CLI at the offline sample frame and a temporary cache."""

    monkeypatch.setattr(data_mod, "load_events", lambda rebuild=False: sample_events)
    monkeypatch.setattr(data_mod, "fetch_messi_matches", lambda force=False: pl.DataFrame())
    monkeypatch.setattr(data_mod, "fetch_all_messi_events", lambda *a, **k: sample_events)
    return tmp_path


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_dunder_main_entry_point() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "lionelmessi", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_summary(cli_env: Path) -> None:
    result = runner.invoke(app, ["summary"])
    assert result.exit_code == 0
    assert "Goals" in result.stdout


def test_season(cli_env: Path) -> None:
    result = runner.invoke(app, ["season", "2011/2012"])
    assert result.exit_code == 0
    assert "2011/20" in result.stdout


def test_season_unknown_is_user_error(cli_env: Path) -> None:
    result = runner.invoke(app, ["season", "1900/1901"])
    assert result.exit_code == 1


def test_fetch(cli_env: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["--cache-dir", str(tmp_path), "fetch"])
    assert result.exit_code == 0
    assert "Cached" in result.stdout


def test_common_options_accepted_after_command(cli_env: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["fetch", "--no-cache", "--verbose"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["summary", "--no-cache", "--verbose"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["xt-plot", "--model", str(tmp_path / "nope.npz"), "--verbose"])
    assert result.exit_code == 1  # option parsed; the model is genuinely missing
    result = runner.invoke(app, ["version", "--no-cache"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_fetch_offline_degrades_to_exit_2(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    def boom(*args: object, **kwargs: object) -> object:
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(data_mod, "fetch_messi_matches", boom)
    monkeypatch.setattr(data_mod, "fetch_all_messi_events", boom)
    result = runner.invoke(app, ["fetch"])
    assert result.exit_code == 2
    assert "Network error" in result.stdout


def test_shotmap(cli_env: Path, tmp_path: Path) -> None:
    out = tmp_path / "shots.png"
    result = runner.invoke(app, ["shotmap", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_goalmap(cli_env: Path, tmp_path: Path) -> None:
    out = tmp_path / "goals.png"
    result = runner.invoke(app, ["goalmap", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_goaltmap_alias(cli_env: Path, tmp_path: Path) -> None:
    out = tmp_path / "goals2.png"
    result = runner.invoke(app, ["goaltmap", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_xt_fit_and_plot(cli_env: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "xt.npz"
    result = runner.invoke(app, ["xt-fit", "--out", str(model_path)])
    assert result.exit_code == 0
    assert model_path.exists()

    plot_path = tmp_path / "xt.png"
    result = runner.invoke(app, ["xt-plot", "--model", str(model_path), "--out", str(plot_path)])
    assert result.exit_code == 0
    assert plot_path.exists()


def test_xt_plot_missing_model_is_user_error(cli_env: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["xt-plot", "--model", str(tmp_path / "nope.npz")])
    assert result.exit_code == 1


def test_continuations(cli_env: Path, tmp_path: Path) -> None:
    out = tmp_path / "chains.csv"
    result = runner.invoke(app, ["continuations", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert pl.read_csv(out).height >= 1


def test_goal(cli_env: Path, tmp_path: Path) -> None:
    out = tmp_path / "goal.png"
    result = runner.invoke(app, ["goal", "evt-005", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_goal_unknown_id_is_user_error(cli_env: Path) -> None:
    result = runner.invoke(app, ["goal", "does-not-exist"])
    assert result.exit_code == 1


def test_report_career(cli_env: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "StatsBomb" in out.read_text(encoding="utf-8")


def test_report_match(cli_env: Path, tmp_path: Path) -> None:
    out = tmp_path / "match.html"
    result = runner.invoke(app, ["report", "--match", "900001", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
