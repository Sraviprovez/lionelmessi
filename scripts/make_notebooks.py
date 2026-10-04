"""Generate the example notebooks under ``notebooks/``.

Run with ``python scripts/make_notebooks.py`` from the project root. The
notebooks are self-contained and runnable once ``lionelmessi fetch`` has warmed
the cache.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOK_1 = notebook(
    [
        md("# 01 — Data exploration\n\nMatches per season, minute distribution, and opponents."),
        code(
            """
import polars as pl
import lionelmessi as lm

events = lm.load_events()
matches = lm.fetch_messi_matches()
matches.group_by("season_name").len().sort("season_name")
"""
        ),
        code(
            """
# Minutes in which Messi acts, and how often he features
pl.concat([
    events.filter(pl.col("player_id") == lm.config.MESSI_PLAYER_ID).select("minute"),
]).describe()
"""
        ),
        code(
            """
# Action types breakdown
events.filter(pl.col("player_id") == lm.config.MESSI_PLAYER_ID) \\
    .group_by("type").len().sort("len", descending=True).head(15)
"""
        ),
        code(
            """
# Heatmap of Messi's touches
ax = lm.viz.plot_heatmap(lm.metrics.messi_only(events))
lm.viz.save_plot(ax, "01_touch_heatmap.png")
"""
        ),
    ]
)

NOTEBOOK_2 = notebook(
    [
        md("# 02 — Expected Threat surface\n\nFit xT on observed transitions and visualise the 'Messi zone'."),
        code(
            """
import lionelmessi as lm

events = lm.load_events()
xt = lm.ExpectedThreat(x_bins=16, y_bins=12).fit(events)
xt.xt_surface.shape
"""
        ),
        code(
            """
ax = xt.plot(annotate=True)
lm.viz.save_plot(ax, "02_xt_surface.png")
"""
        ),
        code(
            """
# The right half-space where Messi created most danger
for xy in [(100, 60), (100, 40), (80, 60), (60, 40)]:
    print(xy, round(xt.xt_at(*xy), 4))
"""
        ),
        code("xt.save('xt.npz')"),
    ]
)

NOTEBOOK_3 = notebook(
    [
        md("# 03 — Goal breakdown\n\nReconstruct the possession chains behind iconic goals."),
        code(
            """
import polars as pl
import lionelmessi as lm

events = lm.load_events()
goals = events.filter((pl.col("is_goal")) & (pl.col("player_id") == lm.config.MESSI_PLAYER_ID))
goal_ids = goals["id"].to_list()
goal_ids[:5]
"""
        ),
        code(
            """
# Visualise the first three goals
for gid in goal_ids[:3]:
    ax = lm.viz.plot_goal_breakdown(events, gid)
    lm.viz.save_plot(ax, f"03_goal_{gid}.png")
"""
        ),
        code(
            """
breakdown = lm.metrics.goal_breakdown(events, goal_ids[0])
print(breakdown["scorer"], breakdown["minute"], breakdown["xg"])
for step in breakdown["chain"]:
    print(step["minute"], step["type"], step["player"])
"""
        ),
    ]
)

NOTEBOOK_4 = notebook(
    [
        md("# 04 — Continuation chains\n\nThe highest-threat possessions ending in a goal."),
        code(
            """
import lionelmessi as lm

events = lm.load_events()
xt = lm.ExpectedThreat().fit(events)
chains = lm.build_continuations(events, xt_model=xt)
big = lm.filter_continuations(chains, actor="Messi", min_xt_gain=0.05, ended_in_goal=True)
summary = lm.continuation_summary(big)
summary.head(10)
"""
        ),
        code(
            """
# Plot the best chain
best = sorted(big, key=lambda c: c.xt_gain, reverse=True)[0]
ax = lm.viz.plot_continuation(best)
lm.viz.save_plot(ax, "04_best_chain.png")
"""
        ),
        code("summary.write_csv('04_chains.csv')"),
    ]
)


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for name, nb in {
        "01_data_exploration.ipynb": NOTEBOOK_1,
        "02_xt_surface.ipynb": NOTEBOOK_2,
        "03_goal_breakdown.ipynb": NOTEBOOK_3,
        "04_continuation_chains.ipynb": NOTEBOOK_4,
    }.items():
        (NOTEBOOKS / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
