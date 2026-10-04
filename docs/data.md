# Data

## Source

All data comes from the **StatsBomb Open Data** repository. It is free for
non-commercial use; review StatsBomb's terms before using the outputs. This
project bundles no proprietary data and only downloads open files on demand.

## Cache layout

Everything is cached under `~/.cache/lionelmessi` (override with the
`LM10_CACHE_DIR` environment variable or `--cache-dir`):

```
~/.cache/lionelmessi/
├── competitions.json
├── matches.parquet
├── lineups/<match_id>.json
├── events/<match_id>.parquet
└── events_all.parquet
```

Once warm, every function runs offline. The cache is deterministic: identical
inputs yield identical outputs.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LM10_CACHE_DIR` | `~/.cache/lionelmessi` | Cache directory |
| `LM10_SB_BASE` | StatsBomb raw URL | Data root |
| `LM10_TIMEOUT` | `30` | HTTP timeout (s) |
| `LM10_MAX_WORKERS` | `8` | Concurrent downloads |
| `LM10_REQUEST_DELAY_MS` | `100` | Delay between batches (ms) |

## Matching Messi

Matches are detected from the teams Messi has represented (Barcelona,
Paris Saint-Germain, Argentina) and then verified against the confirmed lineup
data using player id `5503`. Events are flattened into a tidy columnar frame and
augmented with boolean flags such as `is_goal`, `is_assist`, `is_key_pass`, and
`is_progressive`. The `is_messi` flag marks the events attributable to Messi,
matching on player id `5503` or any of the known name spellings in
`config.MESSI_NAME_CANDIDATES`.

## Offline usage

Once the cache is warm, nothing in the package touches the network: every
command reads from the local Parquet cache. When the network is unavailable and
the cache is cold, `lionelmessi fetch` (and every data-loading command) fails
gracefully with exit code `2` and a clear message instead of a traceback. The
test suite runs fully offline against a small synthetic StatsBomb-like fixture
(`tests/data/sample_events.parquet`) and mocked HTTP responses, so `pytest`
never requires connectivity either.
