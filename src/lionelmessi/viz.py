"""Pitch-level visualizations.

Every plotting function accepts an optional ``ax`` and returns the axes it drew
on.  If :mod:`mplsoccer` is installed it is used for pitch geometry; otherwise a
native Matplotlib pitch is drawn so the package works without the ``viz`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from lionelmessi import config  # noqa: E402
from lionelmessi.metrics import rolling_form  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from lionelmessi.models import ExpectedThreat

__all__ = [
    "save_plot",
    "plot_pitch",
    "plot_xt_surface",
    "plot_shot_map",
    "plot_goal_map",
    "plot_assist_map",
    "plot_pass_map",
    "plot_pass_network",
    "plot_carry_map",
    "plot_heatmap",
    "plot_continuation",
    "plot_goal_breakdown",
    "plot_career_timeline",
    "plot_rolling_form",
    "plot_season_heatmap",
]

PALETTE = config.LM10_PALETTE
_LENGTH = config.PITCH_LENGTH
_WIDTH = config.PITCH_WIDTH


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_cmap(name: str) -> Any:
    """Return ``name`` if Matplotlib knows it, else a seaborn fallback."""

    if name in matplotlib.colormaps:
        return name
    try:  # optional seaborn colormaps (e.g. "rocket")
        import seaborn as sns  # noqa: PLC0415

        return sns.color_palette(name, as_cmap=True)
    except Exception:  # noqa: BLE001 - any failure falls back to Matplotlib
        return "magma"


def _mplsoccer_pitch() -> Any | None:
    """Return the mplsoccer ``Pitch`` class when the ``viz`` extra is installed."""

    try:
        from mplsoccer import Pitch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - exercised only without the extra
        return None
    return Pitch


def _axes(ax: Axes | None, *, figsize: tuple[float, float] = (10.5, 7.0)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def save_plot(ax: Axes, path: str | Path, *, dpi: int = 300) -> Path:
    """Persist the figure owning ``ax`` to ``path``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = cast(Figure, ax.figure)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    return target


def _finish(
    ax: Axes,
    title: str,
    *,
    note: str = config.SB_SOURCE_NOTE,
    xlabel: str = "Pitch length (m)",
    ylabel: str = "Pitch width (m)",
) -> Axes:
    ax.set_title(title, color=PALETTE["text"], fontsize=12, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(
        0.0,
        -0.09,
        note,
        transform=ax.transAxes,
        fontsize=7,
        color="#666666",
        ha="left",
        va="top",
    )
    return ax


def plot_pitch(
    ax: Axes | None = None, *, pitch_type: str = "statsbomb", half: bool = False
) -> Axes:
    """Draw a StatsBomb-oriented pitch (origin at the left goal).

    Prefers :mod:`mplsoccer` geometry when the ``viz`` extra is installed and
    falls back to a pure-Matplotlib pitch otherwise (and for half pitches,
    where the native orientation is kept).
    """

    ax = _axes(ax)
    ax.set_facecolor(PALETTE["pitch"])
    pitch_cls = _mplsoccer_pitch() if pitch_type == "statsbomb" else None
    if pitch_cls is not None and not half:
        pitch = pitch_cls(
            pitch_type="statsbomb",
            pitch_color=PALETTE["pitch"],
            line_color=PALETTE["lines"],
            linewidth=1.4,
            axis=False,
            tick=False,
        )
        pitch.draw(ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    line = PALETTE["lines"]
    lw = 1.4
    ax.set_xlim(0, _LENGTH)
    ax.set_ylim(0, _WIDTH)
    ax.plot(
        [0, 0, _LENGTH, _LENGTH, 0, 0, _LENGTH],
        [0, _WIDTH, _WIDTH, 0, 0, _WIDTH, _WIDTH],
        color=line,
        lw=lw,
    )
    ax.plot([_LENGTH / 2] * 2, [0, _WIDTH], color=line, lw=lw)
    centre = Circle((_LENGTH / 2, _WIDTH / 2), 9.15, color=line, fill=False, lw=lw)
    ax.add_patch(centre)
    ax.add_patch(Circle((_LENGTH / 2, _WIDTH / 2), 0.3, color=line))
    for side in (0.0, _LENGTH):
        sign = 1.0 if side == 0 else -1.0
        ax.plot(
            [side + sign * 18, side + sign * 18, side + sign * 18],
            [(_WIDTH - 44) / 2, (_WIDTH - 44) / 2 + 44, (_WIDTH - 44) / 2],
            color=line,
            lw=lw,
        )
        ax.plot(
            [side, side + sign * 6, side + sign * 6, side],
            [(_WIDTH - 20) / 2, (_WIDTH - 20) / 2, (_WIDTH + 20) / 2, (_WIDTH + 20) / 2],
            color=line,
            lw=lw,
        )
        ax.plot(
            [side + sign * 12] * 2,
            [(_WIDTH - 12) / 2, (_WIDTH + 12) / 2],
            color=line,
            lw=lw,
        )
    if half:
        ax.set_xlim(0, _LENGTH / 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# --------------------------------------------------------------------------- #
# Surfaces & heatmaps
# --------------------------------------------------------------------------- #
def plot_xt_surface(
    xt_model: ExpectedThreat,
    ax: Axes | None = None,
    *,
    cmap: str = "magma",
    annotate: bool = False,
) -> Axes:
    """Render an Expected Threat surface on a pitch."""

    ax = _axes(ax)
    ax = plot_pitch(ax, pitch_type="statsbomb")
    surface = xt_model.xt_surface
    image = ax.imshow(
        surface,
        extent=(0, _LENGTH, 0, _WIDTH),
        origin="lower",
        cmap=cmap,
        aspect="auto",
        alpha=0.85,
        vmin=float(np.min(surface)),
        vmax=float(np.max(surface)),
    )
    ax.figure.colorbar(image, ax=ax, fraction=0.036, pad=0.02, label="Expected threat")
    if annotate:
        x_step = _LENGTH / xt_model.x_bins
        y_step = _WIDTH / xt_model.y_bins
        for xi in range(xt_model.x_bins):
            for yi in range(xt_model.y_bins):
                value = surface[yi, xi]
                ax.text(
                    (xi + 0.5) * x_step,
                    (yi + 0.5) * y_step,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                )
    return _finish(ax, "Expected Threat surface")


def plot_heatmap(
    df: pl.DataFrame,
    ax: Axes | None = None,
    *,
    bins: tuple[int, int] = (16, 12),
    cmap: str = "rocket",
) -> Axes:
    """Render a 2D touch heatmap of ``df`` locations."""

    ax = _axes(ax)
    ax = plot_pitch(ax)
    points = df.drop_nulls(["location_x", "location_y"])
    xs = points["location_x"].to_numpy()
    ys = points["location_y"].to_numpy()
    x_edges = np.linspace(0, _LENGTH, bins[0] + 1)
    y_edges = np.linspace(0, _WIDTH, bins[1] + 1)
    hist, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    image = ax.imshow(
        hist.T,
        extent=(0, _LENGTH, 0, _WIDTH),
        origin="lower",
        cmap=_resolve_cmap(cmap),
        aspect="auto",
        alpha=0.8,
        interpolation="bicubic",
    )
    ax.figure.colorbar(image, ax=ax, fraction=0.036, pad=0.02, label="Touches")
    return _finish(ax, "Touch heatmap")


def plot_season_heatmap(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    """Render a season by pitch-third heatmap of action counts."""

    ax = _axes(ax)
    frame = df.drop_nulls(["location_x"])
    thirds = (
        frame.with_columns(
            pl.when(pl.col("location_x") < 40)
            .then(pl.lit("Defensive"))
            .when(pl.col("location_x") < 80)
            .then(pl.lit("Middle"))
            .otherwise(pl.lit("Final"))
            .alias("third")
        )
        .group_by(["season_name", "third"])
        .len()
    )
    if not thirds.height:
        return _finish(ax, "Season x zone", xlabel="Season", ylabel="Pitch third")
    pivot = thirds.pivot(index="season_name", on="third", values="len").fill_null(0)
    for column in ("Defensive", "Middle", "Final"):
        if column not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(column))
    order = ["Defensive", "Middle", "Final"]
    matrix = pivot.select(order).to_numpy().T
    labels = pivot["season_name"].to_list()
    image = ax.imshow(matrix, aspect="auto", cmap=_resolve_cmap("rocket"))
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)), order)
    ax.figure.colorbar(image, ax=ax, fraction=0.036, pad=0.02, label="Actions")
    return _finish(ax, "Season x zone heatmap", xlabel="Season", ylabel="Pitch third")


# --------------------------------------------------------------------------- #
# Event maps
# --------------------------------------------------------------------------- #
def _point_columns(df: pl.DataFrame) -> tuple[str, str]:
    """Return the ``(x, y)`` column names used by ``df``."""

    if "x" in df.columns and "y" in df.columns:
        return "x", "y"
    return "location_x", "location_y"


def _scatter_outcome(ax: Axes, df: pl.DataFrame, *, size_by: str) -> Any:
    x_col, y_col = _point_columns(df)
    xg_col = "xg" if "xg" in df.columns else "shot_xg"
    if size_by == "xg" and xg_col in df.columns:
        xg = df[xg_col].fill_null(0.0).to_numpy()
        sizes = 30 + 500 * xg
    else:
        sizes = np.full(df.height, 60.0)
    colors = [PALETTE["goal"] if g else PALETTE["shot"] for g in df["is_goal"].to_list()]
    return ax.scatter(
        df[x_col].to_list(), df[y_col].to_list(), s=sizes, c=colors, alpha=0.9, edgecolors="white"
    )


def plot_shot_map(
    df: pl.DataFrame,
    ax: Axes | None = None,
    *,
    size_by: str = "xg",
    color_by: str = "outcome",
) -> Axes:
    """Plot shots scaled by xG and coloured by outcome.

    Accepts either the output of :func:`lionelmessi.metrics.shot_map_data` or a
    raw event frame (in which case only shot events are plotted).
    """

    ax = _axes(ax)
    ax = plot_pitch(ax)
    shots = df.filter(pl.col("is_shot")) if "is_shot" in df.columns else df
    if shots.height:
        _scatter_outcome(ax, shots, size_by=size_by)
    goals = int(shots["is_goal"].fill_null(False).sum()) if shots.height else 0
    return _finish(ax, f"Shot map — {shots.height} shots, {goals} goals")


def plot_goal_map(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot goals only."""

    ax = _axes(ax)
    ax = plot_pitch(ax)
    goals = df.filter(pl.col("is_goal"))
    if goals.height:
        ax.scatter(
            goals["location_x"].to_list(),
            goals["location_y"].to_list(),
            s=140,
            c=PALETTE["goal"],
            edgecolors="white",
            zorder=3,
        )
    return _finish(ax, f"Goal map — {goals.height} goals")


def plot_assist_map(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the origin of every goal assist with an arrow to the recipient."""

    ax = _axes(ax)
    ax = plot_pitch(ax)
    assists = df.filter(pl.col("is_assist"))
    for row in assists.iter_rows(named=True):
        ax.annotate(
            "",
            xy=(row["pass_end_x"], row["pass_end_y"]),
            xytext=(row["location_x"], row["location_y"]),
            arrowprops={"arrowstyle": "-|>", "color": PALETTE["assist"], "lw": 1.6},
        )
        ax.scatter([row["location_x"]], [row["location_y"]], s=40, c=PALETTE["assist"], zorder=3)
    return _finish(ax, f"Assist map — {assists.height} assists")


def plot_pass_map(
    df: pl.DataFrame,
    ax: Axes | None = None,
    *,
    only_completed: bool = True,
    progressive_only: bool = False,
) -> Axes:
    """Plot passes as arrows, optionally limited to completed/progressive ones."""

    ax = _axes(ax)
    ax = plot_pitch(ax)
    passes = df.filter(pl.col("is_pass") & pl.col("pass_end_x").is_not_null())
    if only_completed:
        passes = passes.filter(pl.col("pass_outcome").is_null())
    if progressive_only:
        passes = passes.filter(pl.col("is_progressive").fill_null(False))
    for row in passes.iter_rows(named=True):
        ax.annotate(
            "",
            xy=(row["pass_end_x"], row["pass_end_y"]),
            xytext=(row["location_x"], row["location_y"]),
            arrowprops={"arrowstyle": "-|>", "color": PALETTE["pass"], "lw": 0.8, "alpha": 0.6},
        )
    return _finish(ax, f"Pass map — {passes.height} passes")


def plot_carry_map(
    df: pl.DataFrame,
    ax: Axes | None = None,
    *,
    progressive_only: bool = True,
) -> Axes:
    """Plot carries as arrows."""

    ax = _axes(ax)
    ax = plot_pitch(ax)
    carries = df.filter(pl.col("is_carry") & pl.col("carry_end_x").is_not_null())
    if progressive_only:
        carries = carries.filter(pl.col("is_progressive").fill_null(False))
    for row in carries.iter_rows(named=True):
        ax.annotate(
            "",
            xy=(row["carry_end_x"], row["carry_end_y"]),
            xytext=(row["location_x"], row["location_y"]),
            arrowprops={"arrowstyle": "-|>", "color": PALETTE["carry"], "lw": 1.2, "alpha": 0.8},
        )
    return _finish(ax, f"Carry map — {carries.height} carries")


def plot_pass_network(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    """A coarse pass network: nodes are pitch thirds, edges are pass counts."""

    ax = _axes(ax)
    ax = plot_pitch(ax)
    passes = df.filter(pl.col("is_pass") & pl.col("pass_end_x").is_not_null())
    if not passes.height:
        return _finish(ax, "Pass network — no passes")

    def _zone(x: float) -> int:
        return 0 if x < 40 else (1 if x < 80 else 2)

    edges: dict[tuple[int, int], int] = {}
    for row in passes.iter_rows(named=True):
        a = _zone(row["location_x"])
        b = _zone(row["pass_end_x"])
        edges[(a, b)] = edges.get((a, b), 0) + 1
    max_count = max(edges.values())
    centroids = {0: (20.0, _WIDTH / 2), 1: (60.0, _WIDTH / 2), 2: (100.0, _WIDTH / 2)}
    for (a, b), count in edges.items():
        if a == b:
            continue
        ax.annotate(
            "",
            xy=centroids[b],
            xytext=centroids[a],
            arrowprops={
                "arrowstyle": "-|>",
                "color": PALETTE["pass"],
                "lw": 0.5 + 4 * count / max_count,
                "alpha": 0.5,
            },
        )
    for zone, (x, y) in centroids.items():
        total = sum(c for (a, _), c in edges.items() if a == zone)
        ax.scatter([x], [y], s=120 + total / 5, c=PALETTE["accent"], edgecolors="white", zorder=3)
        ax.text(x, y, str(zone + 1), ha="center", va="center", fontsize=8, color=PALETTE["text"])
    return _finish(ax, "Pass network")


# --------------------------------------------------------------------------- #
# Chains & timelines
# --------------------------------------------------------------------------- #
def plot_continuation(chain: Any, ax: Axes | None = None) -> Axes:
    """Plot a single continuation: arrows for passes/carries, star for shots."""

    ax = _axes(ax)
    ax = plot_pitch(ax)
    for event in chain.events:
        raw = event.raw
        start = (raw.get("location_x"), raw.get("location_y"))
        end_x = raw.get("pass_end_x") or raw.get("carry_end_x")
        end_y = raw.get("pass_end_y") or raw.get("carry_end_y")
        if start[0] is None or end_x is None:
            continue
        color = PALETTE["carry"] if event.type == "Carry" else PALETTE["pass"]
        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4, "alpha": 0.85},
        )
        if event.type == "Shot":
            ax.scatter(
                [start[0]],
                [start[1]],
                marker="*",
                s=260,
                c=PALETTE["goal"],
                edgecolors="white",
                zorder=4,
            )
    title = (
        f"Continuation — {chain.n_events} events, xG {chain.xg:.2f}, "
        f"{'GOAL' if chain.goal else 'no goal'}"
    )
    return _finish(ax, title)


def plot_goal_breakdown(df: pl.DataFrame, goal_event_id: str, ax: Axes | None = None) -> Axes:
    """Visualize the full possession chain that produced a goal."""

    from lionelmessi.metrics import goal_breakdown  # noqa: PLC0415

    breakdown = goal_breakdown(df, goal_event_id)
    ax = _axes(ax)
    ax = plot_pitch(ax)
    previous: tuple[float, float] | None = None
    for step in breakdown["chain"]:
        current = (step["x"], step["y"])
        if previous is not None and previous[0] is not None:
            ax.annotate(
                "",
                xy=current,
                xytext=previous,
                arrowprops={"arrowstyle": "-|>", "color": PALETTE["pass"], "lw": 1.4, "alpha": 0.8},
            )
        if current[0] is not None:
            marker = "*" if step["type"] == "Shot" else "o"
            color = PALETTE["goal"] if step["type"] == "Shot" else PALETTE["accent"]
            size = 260 if step["type"] == "Shot" else 40
            ax.scatter(
                [current[0]],
                [current[1]],
                marker=marker,
                s=size,
                c=color,
                edgecolors="white",
                zorder=3,
            )
        previous = current
    return _finish(
        ax,
        f"Goal breakdown — {breakdown['scorer']} (min {breakdown['minute']}), "
        f"xG {breakdown['xg']:.2f}",
    )


def plot_career_timeline(df: pl.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot cumulative goals and assists over Messi's career."""

    ax = _axes(ax)
    form = rolling_form(df, window=1)
    if not form.height:
        return _finish(ax, "Career timeline", xlabel="Match", ylabel="Cumulative contributions")
    goals = form["goals"].to_numpy()
    assists = form["assists"].to_numpy()
    x = np.arange(len(goals))
    ax.plot(x, np.cumsum(goals), color=PALETTE["goal"], lw=2, label="Goals")
    ax.plot(x, np.cumsum(assists), color=PALETTE["assist"], lw=2, label="Assists")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    return _finish(
        ax,
        "Career timeline — cumulative goals & assists",
        xlabel="Match (chronological)",
        ylabel="Cumulative contributions",
    )


def plot_rolling_form(df: pl.DataFrame, *, window: int = 5, ax: Axes | None = None) -> Axes:
    """Plot goals + assists per match with a rolling mean overlay."""

    ax = _axes(ax)
    form = rolling_form(df, window=window)
    if not form.height:
        return _finish(
            ax,
            f"Rolling form (window={window})",
            xlabel="Match",
            ylabel="Contributions",
        )
    contributions = form["goal_contributions"].to_numpy()
    rolling = form[f"rolling_{window}_contributions"].to_numpy()
    x = np.arange(len(contributions))
    ax.bar(x, contributions, color=PALETTE["pass"], alpha=0.5, label="Per match")
    ax.plot(x, rolling, color=PALETTE["goal"], lw=2, label=f"{window}-match rolling mean")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    return _finish(
        ax,
        f"Rolling form (window={window})",
        xlabel="Match (chronological)",
        ylabel="Goals + assists",
    )
