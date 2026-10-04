"""Regenerate the documentation hero images under ``docs/assets/``.

Runs fully offline against the synthetic StatsBomb-like fixture used by the
test suite, so it reflects the current code without any network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    from lionelmessi import chains, metrics, models, viz
    from tests import sample_data

    events = sample_data.load_or_build()
    assets = ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    ax = viz.plot_shot_map(metrics.shot_map_data(events))
    viz.save_plot(ax, assets / "shot_map.png")

    xt = models.ExpectedThreat().fit(events)
    ax = viz.plot_xt_surface(xt, annotate=True)
    viz.save_plot(ax, assets / "xt_surface.png")

    goal_chains = chains.filter_continuations(
        chains.build_continuations(events, xt_model=xt), ended_in_goal=True
    )
    ax = viz.plot_continuation(goal_chains[0])
    viz.save_plot(ax, assets / "goal_breakdown.png")

    for name in ("shot_map.png", "xt_surface.png", "goal_breakdown.png"):
        size = (assets / name).stat().st_size
        print(f"wrote {name} ({size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
