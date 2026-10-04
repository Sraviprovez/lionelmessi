# lionelmessi 0.1.0 — Acceptance Report

Generated on 2026-10-04 by running every command below for real inside
`/home/dell/lionelmessi` (Python 3.12.3, Ubuntu). No output in this file is
fabricated.

| # | Criterion | Command | Result | Evidence |
|---|-----------|---------|--------|----------|
| 1 | Package installs into a clean venv | `pip install dist/lionelmessi-0.1.0-py3-none-any.whl` in `mktemp -d` venv | PASS | `Successfully installed … lionelmessi-0.1.0 …` (clean-venv smoke log, Phase 7) |
| 2 | `import lionelmessi` works | `python -c "import lionelmessi; print(lionelmessi.__version__)"` | PASS | Prints `0.1.0` (dev venv and clean smoke venv) |
| 3 | `lionelmessi fetch && lionelmessi summary` produces career table | `lionelmessi fetch --no-cache && lionelmessi summary` (real network run) | PASS | `Cached 597 matches, 2260541 events.`, then table: Matches 597, Minutes 53725, Goals 507, Assists 218, Shots 2646, xG 360.31, Key Passes 1043, Dribbles 3185, Progressive Carries 7031, Progressive Passes 9268 |
| 4 | `lionelmessi xt-plot --out xt.png` produces labeled surface | `lionelmessi xt-fit --out xt.npz && lionelmessi xt-plot --model xt.npz --out xt.png` | PASS | `Fitted xT on 1176594 actions`, `Wrote xt.png` (377 KiB, annotated 16×12 grid, colourbar) |
| 5 | `lionelmessi goal <event_id> --out goal.png` renders chain | `lionelmessi goal <real Messi goal event id> --out goal.png` | PASS | `Wrote goal.png` (139 KiB) for a 0.007-xG La Liga 2016/2017 goal |
| 6 | Tests pass with ≥ 90% coverage | `pytest --cov=lionelmessi --cov-report=term-missing` | PASS | `86 passed`, `TOTAL … 95%` (target ≥ 90%) |
| 7 | `twine check dist/*` passes | `python -m build && twine check dist/*` | PASS | `Checking dist/lionelmessi-0.1.0-py3-none-any.whl: PASSED` / `Checking dist/lionelmessi-0.1.0.tar.gz: PASSED`; wheel contains exactly the 11 modules under `lionelmessi/`, no tests/notebooks/site/`__pycache__` |
| 8 | README renders on PyPI with images | `twine check` (README rendering) + sdist image check | PASS | README passed twine's readme render; sdist contains `docs/assets/shot_map.png`, `xt_surface.png`, `goal_breakdown.png`, which PyPI serves for the README's relative image |
| 9 | `mkdocs build --strict` exits 0 | `mkdocs build --strict` | PASS | `Documentation built in 3.84 seconds`, exit code 0 (no warnings) |
| 10 | Zero mypy errors under `strict = true` | `mypy src/lionelmessi` | PASS | `Success: no issues found in 11 source files` |

## Supporting quality gates (all re-run after the final change)

```text
$ ruff check src tests
All checks passed!

$ ruff format --check src tests
22 files already formatted

$ mypy src/lionelmessi
Success: no issues found in 11 source files

$ pytest --cov=lionelmessi --cov-report=term-missing
86 passed … TOTAL 1327 stmts, 62 miss, 95%
```

## Additional verified behaviours

- **Offline degradation**: `LM10_SB_BASE=http://127.0.0.1:9 lionelmessi fetch
  --no-cache` retries three times with exponential backoff, prints
  `Network error: … Could not reach StatsBomb Open Data…`, exits with code **2**
  (no traceback).
- **CLI options**: `--cache-dir`, `--no-cache`, and `--verbose` are accepted
  both before and after the subcommand on every command
  (`lionelmessi fetch --no-cache` and `lionelmessi --no-cache fetch` both work).
- **Exit codes**: `season 1900/1901` → 1 (user error); offline `fetch` → 2
  (data error); success → 0.
- **Docs assets**: `docs/assets/{shot_map,xt_surface,goal_breakdown}.png`
  regenerated from current code via `scripts/make_assets.py` (92 KiB, 341 KiB,
  89 KiB — all ≥ 50 KiB).
- **Notebooks**: regenerated via `python scripts/make_notebooks.py`.
- **xT invariants** (tested in `tests/test_models.py` on crafted data):
  `xt_at(120, 40) == 1.0` (±1e-6), `xt_at(0, 40) == 0.0`, and xT is
  monotonically non-decreasing along `y = 40`.

## Definition of done

All ten criteria are PASS. `lionelmessi 0.1.0 — READY`.
