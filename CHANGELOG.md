# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-04

### Added

- Initial release.
- StatsBomb Open Data ingestion with on-disk Parquet caching, retries, and
  parallel downloads.
- Career, seasonal, competition, and per-90 aggregation metrics.
- Expected Threat (xT) grid model fitted from observed pass and carry
  transitions, with value iteration and `.npz` persistence.
- Possession-chain ("continuation") construction, filtering, and summaries.
- Pitch-level visualizations: pitch, shot map, goal/assist maps, pass map,
  pass network, carry map, heatmap, xT surface, continuation and goal
  breakdown plots, career timeline, and rolling form.
- Auto-generated match, season, and career reports exportable to HTML and PDF.
- `lionelmessi` command-line interface built with Typer and Rich.
