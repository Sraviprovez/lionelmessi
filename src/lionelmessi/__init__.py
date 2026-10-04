"""lionelmessi — scientific football analytics for Lionel Messi's career.

The package ingests StatsBomb Open Data and provides pitch-level spatial
analytics, Expected Threat and VAEP-style action values, possession-chain
analysis, and publication-quality visualizations.

Not affiliated with, endorsed by, or sponsored by Lionel Messi.
"""

from __future__ import annotations

from lionelmessi import chains, config, metrics, models, reports, types, viz
from lionelmessi.chains import build_continuations, continuation_summary, filter_continuations
from lionelmessi.data import (
    fetch_all_messi_events,
    fetch_match_events,
    fetch_messi_matches,
    load_events,
)
from lionelmessi.metrics import (
    assist_breakdown,
    assist_map_data,
    career_summary,
    competition_breakdown,
    goal_breakdown,
    per_90,
    pitch_zone_heatmap,
    progressive_actions,
    rolling_form,
    seasonal_breakdown,
    shot_map_data,
)
from lionelmessi.models import ExpectedThreat

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "fetch_messi_matches",
    "fetch_match_events",
    "fetch_all_messi_events",
    "load_events",
    "career_summary",
    "seasonal_breakdown",
    "competition_breakdown",
    "per_90",
    "shot_map_data",
    "assist_map_data",
    "progressive_actions",
    "pitch_zone_heatmap",
    "goal_breakdown",
    "assist_breakdown",
    "rolling_form",
    "ExpectedThreat",
    "build_continuations",
    "filter_continuations",
    "continuation_summary",
    "viz",
    "metrics",
    "models",
    "chains",
    "reports",
    "config",
    "types",
]
