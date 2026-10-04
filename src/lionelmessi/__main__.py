"""Allow ``python -m lionelmessi`` to invoke the CLI."""

from __future__ import annotations

from lionelmessi.cli import app

if __name__ == "__main__":  # pragma: no cover
    app()
