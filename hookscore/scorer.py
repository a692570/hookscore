"""Core scoring logic for hookscore engagement prediction."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CHUNK_SIZE = 15
DEFAULT_OVERLAP = 3

SCORING_PROMPT = (
    "Someone is reading a social media post. They have read so far:\n\n"
    "{context}\n\n"
    "The next part of the post is:\n\n"
    "{chunk}\n\n"
    "On a scale of 1-10, how engaging/surprising is this next part for the reader?\n"
    "1 = completely predictable, reader skims past it\n"
    "3 = mildly interesting but expected\n"
    "5 = average, holds attention normally\n"
    "7 = surprising, makes reader stop and think\n"
    "10 = jaw-dropping, share immediately\n\n"
    "Reply with ONLY the number."
)


@dataclass
class ChunkResult:
    chunk_idx: int
    text: str
    surprise: float
    level: str = ""
    words: str = ""
    error: Optional[str] = None


@dataclass
class AnalysisReport:
    chunks: list[ChunkResult]
    avg: float = 0.0
    peak: float = 0.0
    peak_idx: int = 0
    trough: float = 0.0
    trough_idx: int = 0
    variance: float = 0.0
    verdict: str = ""
    dropoffs: list[tuple[int, float]] = field(default_factory=list)
    weak_chunks: list[ChunkResult] = field(default_factory=list)


def _get_config(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _score_chunk(
    context: str,
    chunk: str,
    api_base: str,
    model: str,
    api_key: str,
    timeout: float = 20.0,
) -> float:
    """Score a single chunk using LLM-as-judge. Returns 0.1-1.0."""
    prompt = SCORING_PROMPT.format(context=context, chunk=chunk)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.post(
        f"{api_base}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    raw = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    nums = re.findall(r"\b(\d+)\b", raw)
    if nums:
        score = int(nums[-1])
        return max(1, min(10, score)) / 10
    return 0.5


def _level(surprise: float) -> str:
    if surprise > 0.75:
        return "HIGH"
    if surprise > 0.6:
        return "MED-HIGH"
    if surprise > 0.4:
        return "MEDIUM"
    if surprise > 0.25:
        return "LOW"
    return "VERY LOW"


def score_text(
    text: str,
    api_base: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    timeout: float = 20.0,
) -> list[ChunkResult]:
    """Score text sequentially, returning per-chunk results."""
    api_base = api_base or _get_config("HOOKSCORE_API_BASE", DEFAULT_API_BASE)
    model = model or _get_config("HOOKSCORE_MODEL", DEFAULT_MODEL)
    api_key = api_key or _get_config("HOOKSCORE_API_KEY", "")

    words = text.split()
    if len(words) < chunk_size:
        return [ChunkResult(chunk_idx=0, text=text, surprise=0.5, level="baseline", words=f"0-{len(words)}")]

    results: list[ChunkResult] = []
    chunk_starts = list(range(0, len(words), chunk_size - overlap))

    for i, start in enumerate(chunk_starts):
        end = min(start + chunk_size, len(words))
        context_so_far = " ".join(words[:start])
        actual_next = " ".join(words[start:end])

        if not context_so_far:
            results.append(ChunkResult(chunk_idx=i, text=actual_next, surprise=0.5, level="baseline", words=f"{start}-{end}"))
            continue

        try:
            surprise = _score_chunk(context_so_far, actual_next, api_base, model, api_key, timeout)
            results.append(ChunkResult(chunk_idx=i, text=actual_next, surprise=round(surprise, 3), level=_level(surprise), words=f"{start}-{end}"))
        except Exception as e:
            results.append(ChunkResult(chunk_idx=i, text=actual_next, surprise=-1, level="ERROR", words=f"{start}-{end}", error=str(e)[:200]))

    return results


def analyze(results: list[ChunkResult]) -> AnalysisReport:
    """Produce a full analysis report from chunk results."""
    valid = [r for r in results if r.surprise >= 0]
    if not valid:
        return AnalysisReport(chunks=results, verdict="NO VALID RESULTS")

    surprises = [r.surprise for r in valid]
    avg = sum(surprises) / len(surprises)
    peak = max(surprises)
    trough = min(surprises)
    variance = sum((s - avg) ** 2 for s in surprises) / len(surprises)

    if avg > 0.65 and variance > 0.02:
        verdict = "HIGH ENGAGEMENT - surprising + dynamic"
    elif avg > 0.55:
        verdict = "GOOD - keeps reader engaged"
    elif avg > 0.45:
        verdict = "AVERAGE - some strong moments, some weak"
    elif avg < 0.35:
        verdict = "LOW - too predictable, reader may skim"
    elif variance < 0.005:
        verdict = "FLAT - consistent but not dynamic"
    else:
        verdict = "MIXED - needs more tension"

    dropoffs = []
    for i in range(1, len(surprises)):
        drop = surprises[i - 1] - surprises[i]
        if drop > 0.3:
            dropoffs.append((i, drop))

    weak = [r for r in valid if r.surprise < 0.3]

    return AnalysisReport(
        chunks=results,
        avg=avg,
        peak=peak,
        peak_idx=surprises.index(peak),
        trough=trough,
        trough_idx=surprises.index(trough),
        variance=variance,
        verdict=verdict,
        dropoffs=dropoffs,
        weak_chunks=weak,
    )


def format_report(report: AnalysisReport) -> str:
    """Format an AnalysisReport into a printable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("HOOKSCORE - ENGAGEMENT PREDICTION REPORT")
    lines.append("  (LLM Surprise -> Brain Engagement Proxy)")
    lines.append("=" * 60)
    lines.append("")

    for r in report.chunks:
        chunk_text = r.text[:50] + ("..." if len(r.text) > 50 else "")
        if r.surprise >= 0:
            bar_len = int(r.surprise * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            lines.append(f"  [{r.words:>8s}] {r.level:>10s} {bar} {r.surprise:.2f}")
            lines.append(f'       "{chunk_text}"')
        else:
            lines.append(f"  [{r.words:>8s}] {r.level:>10s}")
            lines.append(f'       "{chunk_text}"')
        lines.append("")

    lines.append("-" * 60)
    lines.append(f"  Avg surprise:   {report.avg:.3f}")
    lines.append(f"  Peak surprise:  {report.peak:.3f} (chunk {report.peak_idx})")
    lines.append(f"  Trough:         {report.trough:.3f} (chunk {report.trough_idx})")
    lines.append(f"  Variance:       {report.variance:.4f} (higher = more dynamic)")
    lines.append("")

    for idx, drop in report.dropoffs:
        lines.append(f"  WARNING Drop-off at chunk {idx}: surprise fell {drop:.2f}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  VERDICT: {report.verdict}")

    if report.weak_chunks:
        lines.append("")
        lines.append("  Weakest chunks (rewrite for more surprise):")
        for r in report.weak_chunks:
            lines.append(f'     Chunk {r.chunk_idx}: "{r.text[:60]}..."')

    lines.append("=" * 60)
    return "\n".join(lines)


def compare(
    file_a: str,
    file_b: str,
    api_base: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> str:
    """Compare engagement between two texts."""
    text_a = Path(file_a).read_text() if Path(file_a).exists() else file_a
    text_b = Path(file_b).read_text() if Path(file_b).exists() else file_b

    results_a = score_text(text_a, api_base, model, api_key, chunk_size, overlap)
    results_b = score_text(text_b, api_base, model, api_key, chunk_size, overlap)

    valid_a = [r.surprise for r in results_a if r.surprise >= 0]
    valid_b = [r.surprise for r in results_b if r.surprise >= 0]

    avg_a = sum(valid_a) / len(valid_a) if valid_a else 0
    avg_b = sum(valid_b) / len(valid_b) if valid_b else 0
    var_a = sum((s - avg_a) ** 2 for s in valid_a) / len(valid_a) if valid_a else 0
    var_b = sum((s - avg_b) ** 2 for s in valid_b) / len(valid_b) if valid_b else 0
    peak_a = max(valid_a) if valid_a else 0
    peak_b = max(valid_b) if valid_b else 0

    overall_a = avg_a * 0.6 + min(var_a, 0.05) * 10
    overall_b = avg_b * 0.6 + min(var_b, 0.05) * 10

    lines = []
    lines.append("=" * 50)
    lines.append("A/B ENGAGEMENT COMPARISON")
    lines.append("=" * 50)
    lines.append(f"{'Metric':<25s} {'A':>10s} {'B':>10s} {'Winner':>10s}")
    lines.append("-" * 50)
    lines.append(f"{'Avg surprise':<25s} {avg_a:10.3f} {avg_b:10.3f} {'A' if avg_a > avg_b else 'B':>10s}")
    lines.append(f"{'Variance':<25s} {var_a:10.4f} {var_b:10.4f} {'A' if var_a > var_b else 'B':>10s}")
    lines.append(f"{'Peak surprise':<25s} {peak_a:10.3f} {peak_b:10.3f} {'A' if peak_a > peak_b else 'B':>10s}")
    lines.append("-" * 50)
    winner = "A" if overall_a > overall_b else "B"
    lines.append(f"{'OVERALL':<25s} {overall_a:10.3f} {overall_b:10.3f} {winner:>10s}")
    lines.append("=" * 50)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hookscore",
        description="Score your hook - predict text engagement via LLM surprise",
    )
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--file", "-f", help="Read text from file")
    parser.add_argument("--compare", "-c", nargs=2, metavar=("A", "B"), help="Compare two variants")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Words per chunk (default: 15)")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL (default: https://api.openai.com/v1)")
    parser.add_argument("--model", "-m", default=None, help="Model name (default: gpt-4o-mini)")
    args = parser.parse_args()

    api_base = args.api_base or _get_config("HOOKSCORE_API_BASE", DEFAULT_API_BASE)
    model = args.model or _get_config("HOOKSCORE_MODEL", DEFAULT_MODEL)
    api_key = _get_config("HOOKSCORE_API_KEY", "")

    if args.compare:
        print(compare(args.compare[0], args.compare[1], api_base, model, api_key, args.chunk_size))
        return

    if args.file:
        text = Path(args.file).read_text()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    results = score_text(text, api_base, model, api_key, args.chunk_size)
    report = analyze(results)
    print(format_report(report))


if __name__ == "__main__":
    main()
