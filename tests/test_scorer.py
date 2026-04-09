"""Unit tests for hookscore with mocked API."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from hookscore.scorer import (
    AnalysisReport,
    ChunkResult,
    _level,
    analyze,
    compare,
    format_report,
    score_text,
)


# --- Fixtures / Helpers ---

def _mock_response(score: int):
    """Return a mock httpx response with the given score."""
    content = str(score)

    class MockResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {"message": {"content": content, "reasoning_content": None}}
                ]
            }

    return MockResp()


def _mock_post_factory(scores: list[int]):
    """Return a mock for httpx.post that yields scores in order."""
    idx = [0]

    def _post(url, **kwargs):
        i = idx[0]
        idx[0] += 1
        return _mock_response(scores[i] if i < len(scores) else 5)

    return _post


# --- Tests ---

class TestLevel:
    def test_levels(self):
        assert _level(0.8) == "HIGH"
        assert _level(0.65) == "MED-HIGH"
        assert _level(0.5) == "MEDIUM"
        assert _level(0.3) == "LOW"
        assert _level(0.1) == "VERY LOW"


class TestAnalyze:
    def test_empty_results(self):
        report = analyze([])
        assert report.verdict == "NO VALID RESULTS"

    def test_high_engagement(self):
        chunks = [
            ChunkResult(chunk_idx=0, text="a", surprise=0.7, level="HIGH", words="0-15"),
            ChunkResult(chunk_idx=1, text="b", surprise=0.8, level="HIGH", words="12-27"),
        ]
        report = analyze(chunks)
        assert "HIGH ENGAGEMENT" in report.verdict
        assert report.peak == 0.8

    def test_low_engagement(self):
        chunks = [
            ChunkResult(chunk_idx=0, text="a", surprise=0.2, level="VERY LOW", words="0-15"),
            ChunkResult(chunk_idx=1, text="b", surprise=0.25, level="LOW", words="12-27"),
        ]
        report = analyze(chunks)
        assert "LOW" in report.verdict

    def test_dropoff_detection(self):
        chunks = [
            ChunkResult(chunk_idx=0, text="a", surprise=0.8, level="HIGH", words="0-15"),
            ChunkResult(chunk_idx=1, text="b", surprise=0.3, level="LOW", words="12-27"),
        ]
        report = analyze(chunks)
        assert len(report.dropoffs) == 1
        assert report.dropoffs[0][0] == 1  # chunk index

    def test_weak_chunks(self):
        chunks = [
            ChunkResult(chunk_idx=0, text="a", surprise=0.2, level="VERY LOW", words="0-15"),
            ChunkResult(chunk_idx=1, text="b", surprise=0.7, level="HIGH", words="12-27"),
        ]
        report = analyze(chunks)
        assert len(report.weak_chunks) == 1


class TestFormatReport:
    def test_includes_verdict(self):
        report = AnalysisReport(
            chunks=[ChunkResult(chunk_idx=0, text="hi", surprise=0.5, level="MEDIUM", words="0-2")],
            avg=0.5,
            peak=0.5,
            peak_idx=0,
            trough=0.5,
            trough_idx=0,
            variance=0.0,
            verdict="AVERAGE",
        )
        output = format_report(report)
        assert "AVERAGE" in output
        assert "0.50" in output


class TestScoreText:
    @patch("hookscore.scorer.httpx.post")
    def test_scoring_flow(self, mock_post):
        mock_post.side_effect = _mock_post_factory([7, 5, 8])
        text = " ".join(["word"] * 40)
        results = score_text(text, api_key="test-key", api_base="https://api.example.com/v1", model="test-model")
        assert len(results) > 1
        assert all(r.surprise > 0 for r in results if r.level != "baseline")
        assert mock_post.call_count >= 1

    def test_short_text_baseline(self):
        results = score_text("hello world", api_key="k")
        assert len(results) == 1
        assert results[0].surprise == 0.5
        assert results[0].level == "baseline"

    @patch("hookscore.scorer.httpx.post")
    def test_api_error_handled(self, mock_post):
        mock_post.side_effect = Exception("connection error")
        text = " ".join(["word"] * 40)
        results = score_text(text, api_key="k")
        error_chunks = [r for r in results if r.level == "ERROR"]
        assert len(error_chunks) > 0


class TestCompare:
    @patch("hookscore.scorer.httpx.post")
    def test_compare_output(self, mock_post):
        # Alternate between high and low scores for a vs b
        call_count = [0]

        def _post(url, **kwargs):
            call_count[0] += 1
            return _mock_response(8 if call_count[0] % 2 == 1 else 4)

        mock_post.side_effect = _post
        output = compare(
            " ".join(["word"] * 30),
            " ".join(["word"] * 30),
            api_key="test-key",
        )
        assert "A/B ENGAGEMENT COMPARISON" in output
        assert "OVERALL" in output
