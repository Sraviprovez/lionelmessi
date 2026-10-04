"""Shared pytest fixtures.

All tests run fully offline against a small synthetic Parquet fixture that
mimics StatsBomb Open Data.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from lionelmessi import config  # noqa: E402
from tests import sample_data  # noqa: E402


@pytest.fixture(scope="session")
def sample_events() -> pl.DataFrame:
    """A deterministic event frame equivalent to one Messi match."""

    return sample_data.load_or_build()


@pytest.fixture(autouse=True)
def _close_matplotlib_figures() -> Iterator[None]:
    """Close all Matplotlib figures after every test to avoid leaks."""

    yield
    plt.close("all")


@pytest.fixture()
def isolated_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """Point the cache at a temporary directory for the duration of a test."""

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    yield tmp_path
