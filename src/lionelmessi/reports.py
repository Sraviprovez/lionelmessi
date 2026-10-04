"""Auto-generated match, season, and career reports.

Reports are plain dictionaries holding a summary plus a list of Matplotlib
figures.  :func:`export_html` embeds every figure as a base64 PNG so the output
is fully self-contained; :func:`export_pdf` writes a multi-page PDF.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

import polars as pl

from lionelmessi import metrics, viz

__all__ = ["match_report", "season_report", "career_report", "export_html", "export_pdf"]


def _events_frame(events: pl.DataFrame | None) -> pl.DataFrame:
    if events is not None:
        return events
    from lionelmessi.data import load_events  # noqa: PLC0415

    return load_events()


def _figures_for(frame: pl.DataFrame, *, title_prefix: str = "") -> list[tuple[str, Any]]:
    figures: list[tuple[str, Any]] = []
    shots = metrics.shot_map_data(frame)
    if shots.height:
        ax = viz.plot_shot_map(shots)
        figures.append((f"{title_prefix}Shot map", ax.figure))
    goals = frame.filter(pl.col("is_goal")) if frame.height else frame
    if goals.height:
        ax = viz.plot_goal_map(goals)
        figures.append((f"{title_prefix}Goal map", ax.figure))
    if frame.height:
        ax = viz.plot_rolling_form(frame)
        figures.append((f"{title_prefix}Rolling form", ax.figure))
    return figures


def match_report(match_id: int, *, events: pl.DataFrame | None = None) -> dict[str, Any]:
    """Build a report for a single match."""

    frame = _events_frame(events)
    match = frame.filter(pl.col("match_id") == match_id)
    if not match.height:
        raise KeyError(f"No events cached for match {match_id}")
    summary = metrics.career_summary(match)
    return {
        "kind": "match",
        "match_id": match_id,
        "season_name": match["season_name"][0] if match.height else None,
        "summary": summary,
        "figures": _figures_for(match, title_prefix=f"Match {match_id} — "),
    }


def season_report(season: str, *, events: pl.DataFrame | None = None) -> dict[str, Any]:
    """Build a report for a single season (e.g. ``"2011/2012"``)."""

    frame = _events_frame(events)
    season_frame = frame.filter(pl.col("season_name") == season)
    if not season_frame.height:
        raise KeyError(f"No events cached for season {season}")
    return {
        "kind": "season",
        "season": season,
        "summary": metrics.career_summary(season_frame),
        "figures": _figures_for(season_frame, title_prefix=f"{season} — "),
    }


def career_report(*, events: pl.DataFrame | None = None) -> dict[str, Any]:
    """Build a whole-career report."""

    frame = _events_frame(events)
    return {
        "kind": "career",
        "summary": metrics.career_summary(frame),
        "seasonal": metrics.seasonal_breakdown(frame).to_dicts(),
        "competitions": metrics.competition_breakdown(frame).to_dicts(),
        "figures": _figures_for(frame, title_prefix="Career — "),
    }


def _summary_table(summary: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    return f"<table>{rows}</table>"


def export_html(report: dict[str, Any], path: str | Path) -> Path:
    """Write ``report`` to a self-contained HTML file with embedded images."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>lionelmessi {html.escape(str(report.get('kind', 'report')))} report</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:960px}",
        "table{border-collapse:collapse;margin:1rem 0}",
        "th,td{border:1px solid #ccc;padding:.3rem .8rem;text-align:left}",
        "img{max-width:100%;border:1px solid #eee;margin:.5rem 0}",
        "footer{color:#777;font-size:.8rem;margin-top:2rem}",
        "</style></head><body>",
        f"<h1>lionelmessi {html.escape(str(report.get('kind', 'report')))} report</h1>",
    ]
    if "match_id" in report:
        parts.append(f"<p>Match id: {report['match_id']}</p>")
    if "season" in report:
        parts.append(f"<p>Season: {html.escape(str(report['season']))}</p>")
    parts.append("<h2>Summary</h2>")
    parts.append(_summary_table(report.get("summary", {})))
    for title, figure in report.get("figures", []):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append(f'<img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}">')
    parts.append("<footer>Source: StatsBomb Open Data. Not affiliated with Lionel Messi.</footer>")
    parts.append("</body></html>")
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


def export_pdf(report: dict[str, Any], path: str | Path) -> Path:
    """Write ``report`` to a multi-page PDF using Matplotlib's PDF backend."""

    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(target) as pdf:
        for _, figure in report.get("figures", []):
            pdf.savefig(figure, bbox_inches="tight")
        if not report.get("figures"):
            import matplotlib.pyplot as plt  # noqa: PLC0415

            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.1, 0.9, f"lionelmessi {report.get('kind', 'report')} report", fontsize=16)
            y = 0.84
            for key, value in report.get("summary", {}).items():
                fig.text(0.1, y, f"{key}: {value}", fontsize=10)
                y -= 0.03
            pdf.savefig(fig, bbox_inches="tight")
    return target
