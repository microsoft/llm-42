#!/usr/bin/env python3
"""
Parse benchmark metrics directly from log files.

Log files are produced by the benchmark client (sglang / vLLM) and always
exist after a run completes.  This module extracts throughput (and optional
metadata) from them so that plotting scripts never depend on intermediate
artifacts like summary.csv or benchmark_results.jsonl.

Naming conventions
------------------
  sglang : log_{config}_det{ratio}.log   e.g. log_llm42_ws_32_bs_16_det0.1.log
  vLLM   : log_{config}.log              e.g. log_vllm_deterministic.log

The ``Total token throughput (tok/s)`` line printed by both benchmark
clients is used as the authoritative throughput number.
"""

import re
from pathlib import Path

__all__ = ["parse_log_file", "load_logs_from_dir"]

# ---------------------------------------------------------------------------
# Filename patterns
# ---------------------------------------------------------------------------
# sglang: log_{config}_det{ratio}.log
_RE_SGLANG_LOG = re.compile(r"^log_(.+)_det([\d.]+)\.log$")
# vLLM (no det ratio): log_{config}.log
_RE_VLLM_LOG = re.compile(r"^log_(.+)\.log$")

# ---------------------------------------------------------------------------
# Metric patterns inside the log body
# ---------------------------------------------------------------------------
_RE_TOTAL_THROUGHPUT = re.compile(
    r"Total token throughput \(tok/s\):\s+([\d.]+)"
)
_RE_DURATION = re.compile(
    r"Benchmark duration \(s\):\s+([\d.]+)"
)
_RE_INPUT_TOKENS = re.compile(
    r"Total input tokens:\s+(\d+)"
)
_RE_OUTPUT_TOKENS = re.compile(
    r"Total generated tokens:\s+(\d+)"
)


def _parse_filename(name: str) -> tuple[str, float] | None:
    """Extract (config_name, det_ratio) from a log filename.

    For llm42 configs the returned config_name includes the ``_ratio_X``
    suffix (e.g. ``llm42_ws_32_bs_16_ratio_0.1``) so that downstream code
    can tell different ratios apart.

    Returns None if the filename does not match any known pattern.
    """
    m = _RE_SGLANG_LOG.match(name)
    if m:
        config, ratio_str = m.group(1), m.group(2)
        ratio = float(ratio_str)
        if "llm42" in config:
            return f"{config}_ratio_{ratio_str}", ratio
        return config, ratio
    m = _RE_VLLM_LOG.match(name)
    if m:
        return m.group(1), 1.0
    return None


def parse_log_file(path: Path) -> dict | None:
    """Parse a single benchmark log file.

    Returns a dict with keys:
        config_name           str   – e.g. "sglang_non_deterministic"
        deterministic_ratio   float – e.g. 0.1
        throughput            float – total token throughput (tok/s)

    Returns ``None`` if the file cannot be parsed (unrecognised name or
    missing throughput line).
    """
    parsed = _parse_filename(path.name)
    if parsed is None:
        return None
    config_name, det_ratio = parsed

    text = path.read_text(errors="replace")

    # Primary: use the pre-computed throughput printed by the client
    m = _RE_TOTAL_THROUGHPUT.search(text)
    if m:
        throughput = float(m.group(1))
    else:
        # Fallback: compute from tokens / duration
        m_dur = _RE_DURATION.search(text)
        m_in = _RE_INPUT_TOKENS.search(text)
        m_out = _RE_OUTPUT_TOKENS.search(text)
        if m_dur and m_in and m_out:
            duration = float(m_dur.group(1))
            total_in = int(m_in.group(1))
            total_out = int(m_out.group(1))
            throughput = (total_in + total_out) / duration if duration > 0 else 0.0
        else:
            return None

    return {
        "config_name": config_name,
        "deterministic_ratio": det_ratio,
        "throughput": throughput,
    }


def load_logs_from_dir(directory: Path) -> list[dict]:
    """Scan *directory* for ``log_*.log`` files and parse each one.

    Returns a list of dicts (same schema as :func:`parse_log_file`),
    sorted by config name then deterministic ratio.
    """
    results = []
    if not directory.is_dir():
        return results
    for p in sorted(directory.glob("log_*.log")):
        rec = parse_log_file(p)
        if rec is not None:
            results.append(rec)
    results.sort(key=lambda r: (r["config_name"], r["deterministic_ratio"]))
    return results
