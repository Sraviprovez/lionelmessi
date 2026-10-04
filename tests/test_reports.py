"""Tests for the reports module."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from lionelmessi import reports


def test_match_report(sample_events: pl.DataFrame) -> None:
    report = reports.match_report(900001, events=sample_events)
    assert report["kind"] == "match"
    assert report["summary"]["goals"] == 1
    assert report["figures"]


def test_match_report_unknown_raises(sample_events: pl.DataFrame) -> None:
    with pytest.raises(KeyError):
        reports.match_report(42, events=sample_events)


def test_season_report(sample_events: pl.DataFrame) -> None:
    report = reports.season_report("2011/2012", events=sample_events)
    assert report["kind"] == "season"
    assert report["summary"]["shots"] == 2
    with pytest.raises(KeyError):
        reports.season_report("1999/2000", events=sample_events)


def test_career_report(sample_events: pl.DataFrame) -> None:
    report = reports.career_report(events=sample_events)
    assert report["kind"] == "career"
    assert isinstance(report["seasonal"], list)
    assert report["seasonal"][0]["season_name"] == "2011/2012"
    assert isinstance(report["competitions"], list)


def test_export_html_embeds_images(sample_events: pl.DataFrame, tmp_path: Path) -> None:
    report = reports.career_report(events=sample_events)
    out = reports.export_html(report, tmp_path / "out" / "career.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in text
    assert "Not affiliated" in text


def test_export_html_without_figures(tmp_path: Path) -> None:
    out = reports.export_html(
        {"kind": "career", "summary": {"goals": 0}, "figures": []}, tmp_path / "empty.html"
    )
    assert out.exists()


def test_export_pdf(sample_events: pl.DataFrame, tmp_path: Path) -> None:
    report = reports.career_report(events=sample_events)
    pdf = reports.export_pdf(report, tmp_path / "career.pdf")
    assert pdf.exists()
    assert pdf.stat().st_size > 0


def test_export_pdf_without_figures(tmp_path: Path) -> None:
    pdf = reports.export_pdf(
        {"kind": "career", "summary": {"goals": 1}, "figures": []}, tmp_path / "text.pdf"
    )
    assert pdf.exists()
