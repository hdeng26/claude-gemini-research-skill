"""One Gemini Flash interaction, with budget checks and the evidence kept.

Unlike a plain search wrapper, this keeps what validation needs:
  - the search queries the model actually issued (google_search_call steps)
  - per-URL retrieval status from url_context_result steps
  - url_citation annotations
  - token usage and the cost charged to the ledger
The full raw response is returned too, so the caller can archive it.

Tests replace `_raw_create` (or set GEMINI_RESEARCH_FAKE to a JSON file holding
a list of raw responses, consumed in order) so no API call is made.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import common

TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}


class GeminiError(RuntimeError):
    pass


class SchemaRejected(GeminiError):
    """The API refused the response_format (structured output + tools is preview)."""


def _fake_create(body: dict) -> dict:
    path = Path(os.environ["GEMINI_RESEARCH_FAKE"])
    queue = json.loads(path.read_text())
    state = path.with_suffix(path.suffix + ".idx")
    i = int(state.read_text()) if state.exists() else 0
    state.write_text(str(i + 1))
    if i >= len(queue):
        raise GeminiError(f"fake queue exhausted at call {i}")
    item = queue[i]
    if isinstance(item, dict) and "__error__" in item:
        err = item["__error__"]
        if err.get("schema_rejected"):
            raise SchemaRejected(err.get("message", "schema rejected"))
        raise GeminiError(err.get("message", "fake error"))
    return item


def _raw_create(body: dict, timeout: float) -> dict:
    if os.environ.get("GEMINI_RESEARCH_FAKE"):
        return _fake_create(body)
    from google import genai  # imported lazily so tests need no key

    key = common.load_config().get("GEMINI_API_KEY")
    if not key:
        raise GeminiError(f"GEMINI_API_KEY missing: put it in {common.CONFIG}")
    from google.genai import types

    # SDK-level retries off: the SDK backs off on 429/5xx by itself, which made a
    # quota error take 4 minutes. Retries are decided below, once.
    client = genai.Client(api_key=key, http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1)))
    try:
        resp = client.interactions.create(**body, timeout=timeout, automatic_continuation=True)
    except Exception as e:  # map SDK/HTTP errors to retry decisions
        code = getattr(e, "status_code", None) or getattr(e, "code", None)
        msg = str(e)
        if code == 429 or "429" in msg[:40]:
            if re.search(r"quota|billing|plan", msg, re.I):
                raise QuotaExhausted(
                    "Google says the API quota for this key is exhausted (429). Not retrying. "
                    "Check the key's project billing / limits at https://ai.dev/rate-limit . "
                    f"Raw: {msg[:300]}") from e
            raise _Transient(msg) from e
        if code == 400 and "response_format" in body and re.search(r"schema|response_format|json", msg, re.I):
            raise SchemaRejected(msg) from e
        if code in TRANSIENT_HTTP or "timed out" in msg.lower() or "timeout" in type(e).__name__.lower():
            raise _Transient(msg) from e
        raise GeminiError(f"{type(e).__name__}: {msg}") from e
    return resp.model_dump(mode="json", by_alias=True, exclude_none=True)


class _Transient(GeminiError):
    pass


class QuotaExhausted(GeminiError):
    """Account/project quota exhausted: retrying cannot help."""


def parse(raw: dict) -> dict:
    """Pull the evidence out of a raw interaction dict."""
    queries, url_calls, url_results, citations, texts, errors = [], [], [], [], [], []
    for step in raw.get("steps") or []:
        t = step.get("type")
        if t == "google_search_call":
            queries += (step.get("arguments") or {}).get("queries") or []
        elif t == "url_context_call":
            url_calls += (step.get("arguments") or {}).get("urls") or []
        elif t == "url_context_result":
            for r in step.get("result") or []:
                url_results.append({"url": r.get("url"), "status": r.get("status")})
            if step.get("is_error"):
                errors.append("url_context_result is_error")
        elif t == "google_search_result" and step.get("is_error"):
            errors.append("google_search_result is_error")
        elif t == "model_output":
            if step.get("error"):
                errors.append(f"model_output error: {step['error'].get('message')}")
            offset = sum(len(x) for x in texts)
            for block in step.get("content") or []:
                if block.get("type") != "text":
                    continue
                for a in block.get("annotations") or []:
                    if a.get("type") == "url_citation":
                        citations.append({
                            "url": a.get("url"), "title": a.get("title"),
                            "start": (a.get("start_index") or 0) + offset,
                            "end": (a.get("end_index") or 0) + offset,
                        })
                texts.append(block.get("text") or "")
                offset += len(block.get("text") or "")
    usage = raw.get("usage") or {}
    counted = sum((g.get("count") or 0) for g in usage.get("grounding_tool_count") or []
                  if g.get("type") == "google_search")
    return {
        "status": raw.get("status"),
        "text": "".join(texts) or raw.get("output_text") or "",
        "search_queries": queries,
        "search_query_count": max(len([q for q in queries if q.strip()]), counted),
        "url_calls": url_calls,
        "url_results": url_results,
        "citations": citations,
        "usage": usage,
        "errors": errors,
    }


def extract_json(text: str):
    """Parse JSON from a structured-output reply, or from a fenced block as fallback."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        return json.loads(m.group(1))
    raise GeminiError("reply contained no parseable JSON")


def interact(prompt: str, *, effort: str, label: str, tools=("google_search", "url_context"),
             schema: dict | None = None, system: str | None = None, timeout: float = 180,
             allow_over_daily: bool = False, archive: Path | None = None,
             reserve_usd: float | None = None) -> dict:
    """Run one interaction. Returns parse() output plus `json`, `cost_usd`, `attempts`."""
    if effort not in common.RESERVE_USD:
        raise ValueError(f"effort must be one of {sorted(common.RESERVE_USD)}")
    common.check_budget(effort, allow_over_daily=allow_over_daily, reserve_usd=reserve_usd,
                        expected_queries=10 if "google_search" in tools else 0)

    body = {
        "model": common.model(),
        "input": prompt,
        "generation_config": {"thinking_level": effort},
        "store": False,
    }
    if tools:
        body["tools"] = [{"type": t} for t in tools]
    if system:
        body["system_instruction"] = system
    if schema:
        body["response_format"] = {"type": "text", "mime_type": "application/json", "schema": schema}

    attempts, last_err, raw = [], None, None
    for attempt in range(2):  # one retry, transient errors only
        t0 = time.time()
        try:
            raw = _raw_create(body, timeout)
            attempts.append({"ok": True, "seconds": round(time.time() - t0, 1)})
            break
        except SchemaRejected as e:
            # Structured output with tools is a preview feature; fall back to
            # asking for fenced JSON and validating it ourselves.
            attempts.append({"ok": False, "error": f"schema rejected: {e}", "seconds": round(time.time() - t0, 1)})
            body.pop("response_format", None)
            body["input"] = prompt + "\n\nReturn ONLY one JSON object in a ```json fenced block matching this schema:\n" + json.dumps(schema)
            last_err = e
            continue
        except _Transient as e:
            attempts.append({"ok": False, "error": str(e), "seconds": round(time.time() - t0, 1)})
            last_err = e
            time.sleep(5)
            continue
    if raw is None:
        raise GeminiError(f"gemini call failed after {len(attempts)} attempts: {last_err}")

    if archive:
        archive.write_text(json.dumps(raw, ensure_ascii=False, indent=1))
    out = parse(raw)
    out["cost_usd"] = round(common.cost_usd(out["usage"]), 5)
    out["attempts"] = attempts
    common.record_spend({
        "label": label, "effort": effort, "model": common.model(),
        "cost_usd": out["cost_usd"], "search_queries": out["search_query_count"],
        "input_tokens": out["usage"].get("total_input_tokens"),
        "output_tokens": out["usage"].get("total_output_tokens"),
        "thought_tokens": out["usage"].get("total_thought_tokens"),
        "status": out["status"],
    })
    if out["status"] != "completed":
        raise GeminiError(f"interaction ended with status={out['status']} errors={out['errors']}")
    out["json"] = None
    if schema:
        out["json"] = extract_json(out["text"])
    return out
