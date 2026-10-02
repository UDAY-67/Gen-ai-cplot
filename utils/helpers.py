"""
Helper Utilities Module
Common string manipulation, JSON parsing, latency tracking, and formatting helpers.
"""

import re
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager


def clean_text(text: str) -> str:
    """Normalize whitespace, trim individual lines, and remove unwanted control characters."""
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Rough heuristic: 1 token ≈ 4 characters for English text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract and parse a JSON object from text,
    even if wrapped in markdown code blocks or containing surrounding commentary.
    """
    if not raw_text:
        return None

    # First attempt: Direct json parse
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Second attempt: Look for markdown code block ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Third attempt: Find outermost curly braces { ... }
    brace_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return None


@contextmanager
def measure_latency():
    """Context manager to measure execution latency in seconds."""
    start = time.perf_counter()
    metrics = {"elapsed_seconds": 0.0}
    try:
        yield metrics
    finally:
        metrics["elapsed_seconds"] = round(time.perf_counter() - start, 3)


def format_source_citation(chunk_metadata: Dict[str, Any]) -> str:
    """Format chunk metadata into a clean citation string."""
    doc_name = chunk_metadata.get("doc_name", "Unknown Document")
    page_num = chunk_metadata.get("page_number", "?")
    chunk_id = chunk_metadata.get("chunk_id", "")
    return f"📄 **{doc_name}** — Page {page_num} *(Chunk #{chunk_id})*"
