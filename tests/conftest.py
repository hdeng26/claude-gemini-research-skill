import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import common  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own ledger/runs dir and a fake key; no real API."""
    monkeypatch.setattr(common, "HOME", tmp_path)
    monkeypatch.setattr(common, "CONFIG", tmp_path / "config.env")
    monkeypatch.setattr(common, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(common, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(common, "CACHE", tmp_path / "cache")
    for k in ("MONTHLY_CAP_USD", "DAILY_SOFT_CAP_USD", "MONTHLY_SEARCH_QUERY_CAP", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("GEMINI_RESEARCH_FAKE", raising=False)
    (tmp_path / "config.env").write_text("GEMINI_API_KEY=test-not-real\n")
    return tmp_path


def raw_response(text, queries=("q1",), url_results=(), citations=(), status="completed",
                 usage=None):
    steps = []
    if queries:
        steps.append({"type": "google_search_call", "id": "s1", "arguments": {"queries": list(queries)}})
        steps.append({"type": "google_search_result", "call_id": "s1", "result": [{"search_suggestions": "<div/>"}]})
    if url_results:
        steps.append({"type": "url_context_call", "id": "u1", "arguments": {"urls": [u for u, _ in url_results]}})
        steps.append({"type": "url_context_result", "call_id": "u1",
                      "result": [{"url": u, "status": s} for u, s in url_results]})
    steps.append({"type": "model_output", "content": [{"type": "text", "text": text, "annotations": [
        {"type": "url_citation", "url": u, "title": t, "start_index": a, "end_index": b}
        for u, t, a, b in citations]}]})
    return {"status": status, "steps": steps, "usage": usage or {
        "total_input_tokens": 20000, "total_output_tokens": 1500, "total_thought_tokens": 500,
        "total_tool_use_tokens": 3000, "total_cached_tokens": 0,
        "grounding_tool_count": [{"type": "google_search", "count": len(queries)}]}}
