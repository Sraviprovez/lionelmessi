"""Global configuration for :mod:`lionelmessi`.

All tunables can be overridden through environment variables so that the
package is deterministic (same inputs -> same outputs) and offline-friendly
(once the cache is warm, no network access is required).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "CACHE_DIR",
    "SB_BASE",
    "TIMEOUT",
    "MAX_WORKERS",
    "REQUEST_DELAY_MS",
    "BATCH_DELAY",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "PITCH_HALF_LENGTH",
    "MESSI_PLAYER_ID",
    "MESSI_TEAM_IDS",
    "MESSI_NAME_CANDIDATES",
    "MESSI_COMPETITION_IDS",
    "LM10_PALETTE",
    "SB_SOURCE_NOTE",
    "cache_dir",
]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


#: Directory where Parquet caches and fitted models are written.
CACHE_DIR: Path = _env_path("LM10_CACHE_DIR", Path.home() / ".cache" / "lionelmessi")

#: Root of the StatsBomb Open Data repository (raw files).
SB_BASE: str = os.environ.get(
    "LM10_SB_BASE",
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data",
)

#: HTTP timeout, in seconds.
TIMEOUT: int = _env_int("LM10_TIMEOUT", 30)

#: Maximum number of concurrent download workers.
MAX_WORKERS: int = _env_int("LM10_MAX_WORKERS", 8)

#: Politeness delay between concurrent batches, in milliseconds.
REQUEST_DELAY_MS: int = _env_int("LM10_REQUEST_DELAY_MS", 100)

#: Politeness delay between concurrent batches, in seconds (derived from
#: :data:`REQUEST_DELAY_MS`).
BATCH_DELAY: float = REQUEST_DELAY_MS / 1000.0

#: StatsBomb pitch dimensions (metres).
PITCH_LENGTH: float = 120.0
PITCH_WIDTH: float = 80.0
PITCH_HALF_LENGTH: float = PITCH_LENGTH / 2.0

#: StatsBomb identifier for Lionel Andrés Messi Cuccittini.
MESSI_PLAYER_ID: int = 5503

#: Team ids Messi has represented in StatsBomb Open Data, mapped to a label.
MESSI_TEAM_IDS: dict[int, str] = {
    217: "Barcelona",
    131: "Paris Saint-Germain",
    779: "Argentina",
}

#: Name spellings that identify Messi across StatsBomb editions.
MESSI_NAME_CANDIDATES: tuple[str, ...] = (
    "Lionel Andrés Messi Cuccittini",
    "Lionel Andrés Messi",
    "Lionel Messi",
    "L. Messi",
)

#: Competitions worth scanning for Messi appearances.
MESSI_COMPETITION_IDS: frozenset[int] = frozenset(
    {
        11,  # La Liga
        16,  # Champions League
        87,  # Copa del Rey
        7,  # Ligue 1
        44,  # Major League Soccer
        43,  # FIFA World Cup
        223,  # Copa America
        81,  # Liga Profesional
    }
)

#: Shared colour palette used by every visualization.
LM10_PALETTE: dict[str, str] = {
    "pitch": "#101820",
    "lines": "#ffffff",
    "grass": "#1f8a54",
    "goal": "#e63946",
    "shot": "#f4a261",
    "pass": "#4cc9f0",
    "carry": "#c77dff",
    "assist": "#2a9d8f",
    "xt_low": "#0b525b",
    "xt_high": "#ffdd00",
    "text": "#1b1b1b",
    "accent": "#ffb703",
}

#: Attribution string appended to every figure.
SB_SOURCE_NOTE: str = "Source: StatsBomb Open Data"


def cache_dir(*, override: str | Path | None = None) -> Path:
    """Return the effective cache directory, creating it if necessary.

    Parameters
    ----------
    override:
        Optional explicit directory that takes precedence over
        :data:`CACHE_DIR`. Environment-based configuration still applies when
        ``override`` is ``None``.
    """

    path = Path(override).expanduser() if override is not None else CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path
