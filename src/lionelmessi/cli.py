"""Command-line interface for :mod:`lionelmessi`.

Exit codes: ``0`` success, ``1`` user error, ``2`` data error.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import polars as pl
import requests
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from lionelmessi import __version__, chains, config, metrics, models, viz
from lionelmessi import data as data_mod

app = typer.Typer(
    name="lionelmessi",
    help="Scientific football analytics for Lionel Messi's career.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

_STATE: dict[str, object] = {"cache_dir": None, "no_cache": False}

CacheDirOpt = Annotated[
    Path | None, typer.Option("--cache-dir", help="Override the cache directory.")
]
NoCacheOpt = Annotated[
    bool | None, typer.Option("--no-cache", help="Ignore the cache and re-fetch.")
]
VerboseOpt = Annotated[bool | None, typer.Option("--verbose", "-v", help="Enable verbose logging.")]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
        force=True,
    )


def _apply_common(cache_dir: Path | None, no_cache: bool | None, verbose: bool | None) -> None:
    """Merge command-level overrides into the shared CLI state."""

    if cache_dir is not None:
        _STATE["cache_dir"] = cache_dir
        config.cache_dir(override=cache_dir)
    if no_cache is not None:
        _STATE["no_cache"] = no_cache
    if verbose is not None:
        _configure_logging(verbose)


@app.callback()
def main(
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Configure global options shared by every command."""

    _apply_common(cache_dir, no_cache, verbose)
    _configure_logging(verbose is True)


def _load(rebuild: bool = False) -> pl.DataFrame:
    """Load events, translating data errors into exit code 2."""

    try:
        return data_mod.load_events(rebuild=rebuild or bool(_STATE["no_cache"]))
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        console.print(f"[red]Data error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _summary_table(summary: dict[str, object]) -> Table:
    table = Table(title="Lionel Messi — career summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    for key, value in summary.items():
        table.add_row(str(key).replace("_", " ").title(), str(value))
    return table


@app.command()
def fetch(
    cache_dir: CacheDirOpt = None, no_cache: NoCacheOpt = None, verbose: VerboseOpt = None
) -> None:
    """Ingest and cache every available Messi match and event."""

    _apply_common(cache_dir, no_cache, verbose)
    try:
        with console.status("Fetching StatsBomb Open Data..."):
            matches = data_mod.fetch_messi_matches(force=bool(_STATE["no_cache"]))
            events = data_mod.fetch_all_messi_events(force=bool(_STATE["no_cache"]))
    except (requests.RequestException, OSError) as exc:
        console.print(
            f"[red]Network error:[/red] {exc}\n"
            "Could not reach StatsBomb Open Data. Check your connection and retry, "
            "or run any cache-backed command once the cache is warm."
        )
        raise typer.Exit(code=2) from exc
    combined = config.cache_dir() / "events_all.parquet"
    events.write_parquet(combined)
    console.print(f"[green]Cached[/green] {matches.height} matches, {events.height} events.")


@app.command()
def summary(
    cache_dir: CacheDirOpt = None, no_cache: NoCacheOpt = None, verbose: VerboseOpt = None
) -> None:
    """Print Messi's career summary."""

    _apply_common(cache_dir, no_cache, verbose)
    frame = _load()
    console.print(_summary_table(metrics.career_summary(frame)))


@app.command()
def season(
    name: Annotated[str, typer.Argument(help="Season, e.g. 2011/2012")],
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Print a seasonal breakdown for one season."""

    _apply_common(cache_dir, no_cache, verbose)
    frame = _load()
    breakdown = metrics.seasonal_breakdown(frame).filter(pl.col("season_name") == name)
    if not breakdown.height:
        console.print(f"[yellow]No data for season {name!r}.[/yellow]")
        raise typer.Exit(code=1)
    _print_frame(breakdown)


@app.command()
def shotmap(
    season_name: Annotated[str | None, typer.Option("--season", help="Filter by season.")] = None,
    out: Annotated[Path, typer.Option("--out", help="Output PNG path.")] = Path("shots.png"),
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Render a shot map."""

    _apply_common(cache_dir, no_cache, verbose)
    frame = _load()
    if season_name:
        frame = frame.filter(pl.col("season_name") == season_name)
    ax = viz.plot_shot_map(metrics.shot_map_data(frame))
    viz.save_plot(ax, out)
    console.print(f"[green]Wrote[/green] {out}")


@app.command()
def goalmap(
    out: Annotated[Path, typer.Option("--out", help="Output PNG path.")] = Path("goals.png"),
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Render a goal map."""

    _apply_common(cache_dir, no_cache, verbose)
    frame = _load()
    ax = viz.plot_goal_map(frame.filter(pl.col("is_goal")))
    viz.save_plot(ax, out)
    console.print(f"[green]Wrote[/green] {out}")


@app.command("goaltmap", hidden=True)
def goaltmap(
    out: Annotated[Path, typer.Option("--out", help="Output PNG path.")] = Path("goals.png"),
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Alias of ``goalmap`` kept for CLI compatibility."""

    _apply_common(cache_dir, no_cache, verbose)
    goalmap(out=out)


@app.command()
def xt_fit(
    out: Annotated[Path, typer.Option("--out", help="Output .npz path.")] = Path("xt.npz"),
    x_bins: Annotated[int, typer.Option(help="Grid columns.")] = 16,
    y_bins: Annotated[int, typer.Option(help="Grid rows.")] = 12,
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Fit and persist an Expected Threat model."""

    _apply_common(cache_dir, no_cache, verbose)
    frame = _load()
    model = models.ExpectedThreat(x_bins=x_bins, y_bins=y_bins).fit(frame)
    model.save(out)
    console.print(f"[green]Fitted[/green] xT on {model.n_events} actions -> {out}")


@app.command()
def xt_plot(
    model_path: Annotated[Path, typer.Option("--model", help="Fitted .npz model.")] = Path(
        "xt.npz"
    ),
    out: Annotated[Path, typer.Option("--out", help="Output PNG path.")] = Path("xt.png"),
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Plot a fitted Expected Threat surface."""

    _apply_common(cache_dir, no_cache, verbose)
    if not model_path.exists():
        console.print(f"[red]Model not found:[/red] {model_path}")
        raise typer.Exit(code=1)
    model = models.ExpectedThreat.load(model_path)
    ax = model.plot(annotate=True)
    viz.save_plot(ax, out)
    console.print(f"[green]Wrote[/green] {out}")


@app.command()
def continuations(
    min_xt: Annotated[float, typer.Option("--min-xt", help="Minimum xT gain.")] = 0.0,
    out: Annotated[Path, typer.Option("--out", help="Output CSV path.")] = Path("chains.csv"),
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Build possession chains and write a summary CSV."""

    _apply_common(cache_dir, no_cache, verbose)
    frame = _load()
    built = chains.build_continuations(frame)
    filtered = chains.filter_continuations(built, min_xt_gain=min_xt)
    table_frame = chains.continuation_summary(filtered)
    out.parent.mkdir(parents=True, exist_ok=True)
    table_frame.write_csv(out)
    console.print(f"[green]Wrote[/green] {table_frame.height} chains -> {out}")


@app.command()
def goal(
    event_id: Annotated[str, typer.Argument(help="Goal event id.")],
    out: Annotated[Path, typer.Option("--out", help="Output PNG path.")] = Path("goal.png"),
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Render the full possession chain leading to a goal."""

    _apply_common(cache_dir, no_cache, verbose)
    frame = _load()
    try:
        ax = viz.plot_goal_breakdown(frame, event_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    viz.save_plot(ax, out)
    console.print(f"[green]Wrote[/green] {out}")


@app.command()
def report(
    match: Annotated[int | None, typer.Option("--match", help="Match id.")] = None,
    season_name: Annotated[str | None, typer.Option("--season", help="Season name.")] = None,
    out: Annotated[Path, typer.Option("--out", help="Output HTML path.")] = Path("report.html"),
    cache_dir: CacheDirOpt = None,
    no_cache: NoCacheOpt = None,
    verbose: VerboseOpt = None,
) -> None:
    """Generate an HTML report for a match, season, or the whole career."""

    _apply_common(cache_dir, no_cache, verbose)
    from lionelmessi import reports  # noqa: PLC0415

    frame = _load()
    try:
        if match is not None:
            built = reports.match_report(match, events=frame)
        elif season_name is not None:
            built = reports.season_report(season_name, events=frame)
        else:
            built = reports.career_report(events=frame)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    reports.export_html(built, out)
    console.print(f"[green]Wrote[/green] {out}")


@app.command()
def version(
    cache_dir: CacheDirOpt = None, no_cache: NoCacheOpt = None, verbose: VerboseOpt = None
) -> None:
    """Print the package version."""

    _apply_common(cache_dir, no_cache, verbose)
    console.print(f"lionelmessi {__version__}")


def _print_frame(frame: pl.DataFrame) -> None:
    table = Table(show_header=True)
    for column in frame.columns:
        table.add_column(str(column))
    for row in frame.iter_rows():
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
