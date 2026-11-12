import datetime as dt
import json
import os

import pytest

import common
import gemini_client
import research
import scholar
import validate
from conftest import raw_response


# ---------- parsing -------------------------------------------------------
def test_parse_keeps_queries_url_status_citations():
    raw = raw_response("Paper A [1].", queries=("closed-form layer training", "layerwise least squares"),
                       url_results=(("https://arxiv.org/abs/1", "success"), ("https://x.org/p", "paywall")),
                       citations=(("https://arxiv.org/abs/1", "arxiv.org", 0, 7),))
    p = gemini_client.parse(raw)
    assert p["search_queries"] == ["closed-form layer training", "layerwise least squares"]
    assert p["search_query_count"] == 2
    assert {"url": "https://x.org/p", "status": "paywall"} in p["url_results"]
    assert p["citations"][0]["url"] == "https://arxiv.org/abs/1"
    assert p["text"] == "Paper A [1]."


def test_extract_json_fenced_fallback():
    assert gemini_client.extract_json('blah\n```json\n{"papers": []}\n```') == {"papers": []}
    with pytest.raises(gemini_client.GeminiError):
        gemini_client.extract_json("no json here")


# ---------- cost & budget -------------------------------------------------
def test_cost_is_conservative_and_price_doubles_2027():
    u = {"total_input_tokens": 1_000_000, "total_output_tokens": 1_000_000, "total_thought_tokens": 0}
    assert common.cost_usd(u, dt.date(2026, 11, 11)) == pytest.approx(0.75 + 3.75)
    assert common.cost_usd(u, dt.date(2027, 1, 2)) == pytest.approx(1.50 + 7.50)
    # thought and tool tokens are added on top (overestimate rather than under)
    u2 = {"total_input_tokens": 0, "total_output_tokens": 0, "total_thought_tokens": 1_000_000,
          "total_tool_use_tokens": 1_000_000}
    assert common.cost_usd(u2, dt.date(2026, 11, 11)) == pytest.approx(3.75 + 0.75)


def test_monthly_cap_refuses(isolated_home):
    common.record_spend({"label": "x", "cost_usd": 14.97, "search_queries": 1})
    with pytest.raises(common.BudgetError, match="monthly cap"):
        common.check_budget("low", allow_over_daily=True)


def test_daily_soft_cap_needs_explicit_override(isolated_home):
    common.record_spend({"label": "x", "cost_usd": 1.97, "search_queries": 1})
    with pytest.raises(common.BudgetError, match="daily soft cap"):
        common.check_budget("low")
    common.check_budget("low", allow_over_daily=True)  # user agreed


def test_search_query_cap(isolated_home):
    common.record_spend({"label": "x", "cost_usd": 0, "search_queries": 4495})
    with pytest.raises(common.BudgetError, match="search query cap"):
        common.check_budget("low")


def test_caps_overridable_in_config(isolated_home):
    (isolated_home / "config.env").write_text("GEMINI_API_KEY=k\nMONTHLY_CAP_USD=0.04\n")
    with pytest.raises(common.BudgetError, match="monthly cap"):
        common.check_budget("low")


def test_last_months_spend_does_not_count(isolated_home):
    with open(common.LEDGER, "w") as f:
        f.write(json.dumps({"ts": "2000-01-01T00:00:00", "cost_usd": 99, "search_queries": 9999}) + "\n")
    common.check_budget("high")


# ---------- interact ------------------------------------------------------
def _queue(monkeypatch, items):
    calls = []

    def fake(body, timeout):
        calls.append(json.loads(json.dumps(body)))
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(gemini_client, "_raw_create", fake)
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: None)
    return calls


def test_interact_records_spend_and_sends_effort(monkeypatch):
    calls = _queue(monkeypatch, [raw_response('{"papers": []}')])
    out = gemini_client.interact("p", effort="medium", label="t", schema={"type": "object"})
    assert calls[0]["generation_config"] == {"thinking_level": "medium"}
    assert calls[0]["store"] is False
    assert calls[0]["model"] == "gemini-3.8-flash"
    assert out["json"] == {"papers": []}
    s = common.spend_summary()
    assert s["month_calls"] == 1 and s["month_usd"] > 0 and s["month_search_queries"] == 1


def test_interact_schema_rejected_falls_back_to_fenced_json(monkeypatch):
    calls = _queue(monkeypatch, [gemini_client.SchemaRejected("bad schema"),
                                 raw_response('```json\n{"papers": [{"title": "T"}]}\n```')])
    out = gemini_client.interact("p", effort="low", label="t", schema={"type": "object"})
    assert "response_format" not in calls[1]
    assert out["json"]["papers"][0]["title"] == "T"


def test_interact_transient_retries_once_then_fails(monkeypatch):
    _queue(monkeypatch, [gemini_client._Transient("503"), gemini_client._Transient("503")])
    with pytest.raises(gemini_client.GeminiError, match="after 2 attempts"):
        gemini_client.interact("p", effort="low", label="t")
    assert common.spend_summary()["month_calls"] == 0


def test_interact_non_completed_is_error_but_still_billed(monkeypatch):
    _queue(monkeypatch, [raw_response("partial", status="incomplete")])
    with pytest.raises(gemini_client.GeminiError, match="incomplete"):
        gemini_client.interact("p", effort="low", label="t")
    assert common.spend_summary()["month_calls"] == 1


def test_interact_refused_before_any_call_when_over_budget(monkeypatch, isolated_home):
    calls = _queue(monkeypatch, [raw_response("x")])
    common.record_spend({"label": "x", "cost_usd": 15, "search_queries": 0})
    with pytest.raises(common.BudgetError):
        gemini_client.interact("p", effort="low", label="t")
    assert calls == []


# ---------- validation / scout end-to-end (offline) -----------------------
REAL = {"source": "arxiv", "title": "Attention Is All You Need", "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017, "arxiv_id": "1706.03762", "doi": None, "url": "https://arxiv.org/abs/1706.03762",
        "abstract": "The dominant sequence transduction models...", "venue": "arXiv preprint"}


def fake_resolve(claim):
    if scholar.title_sim(claim.get("title", ""), REAL["title"]) >= 0.9:
        flags = []
        if claim.get("arxiv_id") and scholar.norm_arxiv(claim["arxiv_id"]) != REAL["arxiv_id"]:
            flags.append("claimed_arxiv_id_wrong")
        return {"status": "verified", "record": dict(REAL), "flags": flags, "title_similarity": 1.0,
                "checked": ["arxiv"]}
    return {"status": "unverified", "record": None, "flags": ["no_database_match"],
            "title_similarity": 0.0, "checked": ["openalex:title", "arxiv:title"]}


def _scout_json(papers):
    return json.dumps({"expanded_queries": ["a", "b"], "papers": papers, "notes": "n"})


def test_scout_separates_fabricated_and_overrides_metadata(monkeypatch, isolated_home, capsys):
    monkeypatch.setattr(scholar, "resolve", fake_resolve)
    _queue(monkeypatch, [raw_response(_scout_json([
        {"title": "Attention is all you need", "authors": ["Someone Else"], "year": 2019,
         "arxiv_id": "9999.99999", "relevance": "r"},
        {"title": "attention is all you need.", "relevance": "dup"},
        {"title": "Layerwise Moore-Penrose Transformers Without Backprop", "doi": "10.1234/fake",
         "relevance": "looks perfect"},
    ]), queries=("q1", "q2"))])
    assert research.main(["scout", "layerwise training"]) == 0
    run = next((isolated_home / "runs").iterdir())
    r = json.loads((run / "result.json").read_text())
    assert len(r["papers"]) == 1
    p = r["papers"][0]
    assert p["year"] == 2017 and p["arxiv_id"] == "1706.03762"  # canonical, not Gemini's
    assert "claimed_arxiv_id_wrong" in p["flags"]
    assert p["reading_level"] == "abstract_verified"
    assert p["duplicates_merged"] == 1
    assert r["unverified"][0]["claimed"]["doi"] == "10.1234/fake"
    assert r["search_queries_issued"] == ["q1", "q2"]
    out = capsys.readouterr().out
    assert "Unverified (1)" in out and "do not cite" in out


def test_scout_ungrounded_retries_once_and_warns(monkeypatch, isolated_home):
    monkeypatch.setattr(scholar, "resolve", fake_resolve)
    calls = _queue(monkeypatch, [raw_response(_scout_json([]), queries=()),
                                 raw_response(_scout_json([]), queries=())])
    assert research.main(["scout", "layerwise training"]) == 0
    assert len(calls) == 2 and "MUST call Google Search" in calls[1]["input"]
    r = json.loads(next((isolated_home / "runs").iterdir()).joinpath("result.json").read_text())
    assert any("ungrounded" in f for f in r["grounding_flags"])


def test_scout_budget_refusal_exit_code(monkeypatch, isolated_home):
    common.record_spend({"label": "x", "cost_usd": 15, "search_queries": 0})
    assert research.main(["scout", "layerwise training"]) == 2


def test_no_abstract_means_search_result_only(monkeypatch):
    def resolve_no_abs(c):
        r = fake_resolve(c)
        if r["record"]:
            r["record"]["abstract"] = None
        return r
    monkeypatch.setattr(scholar, "resolve", resolve_no_abs)
    out = validate.validate_papers([{"title": "Attention Is All You Need"}])
    assert out["papers"][0]["reading_level"] == "search_result_only"


# ---------- live metadata (free APIs, opt-in) -----------------------------
@pytest.mark.skipif(not os.environ.get("LIVE_SCHOLAR"), reason="set LIVE_SCHOLAR=1")
def test_live_scholar_real_and_fabricated():
    assert scholar.resolve({"title": "Attention Is All You Need", "arxiv_id": "1706.03762"})["status"] == "verified"
    assert scholar.resolve({"title": "Pseudoinverse Backpropagation-Free Training of Transformers via "
                                     "Layerwise Moore-Penrose Updates"})["status"] == "unverified"
    bad = scholar.resolve({"title": "Attention Is All You Need", "arxiv_id": "1512.03385", "year": 2012})
    assert bad["status"] == "verified" and any("arxiv_id_points_to_other_paper" in f for f in bad["flags"])


# ---------- landing-page fallback ----------------------------------------
def _no_db(monkeypatch):
    for f in ("openalex_search_title", "arxiv_search_title"):
        monkeypatch.setattr(scholar, f, lambda *a, **k: [])


def test_blocked_landing_page_is_unverified_with_manual_flag(monkeypatch):
    _no_db(monkeypatch)
    monkeypatch.setattr(scholar, "landing_page", lambda url: {"blocked": True, "status_code": 403})
    r = scholar.resolve({"title": "Some OpenReview Paper", "url": "https://openreview.net/forum?id=x"})
    assert r["status"] == "unverified"
    assert r["flags"][0].startswith("landing_page_blocked_by_bot_check")


def test_landing_page_title_match_verifies_with_weak_flag(monkeypatch):
    _no_db(monkeypatch)
    rec = {"source": "landing_page", "title": "Some Workshop Paper", "authors": ["A B"], "year": 2025,
           "doi": None, "arxiv_id": None, "url": "https://w.org/p", "abstract": None, "venue": "W"}
    monkeypatch.setattr(scholar, "landing_page", lambda url: {"blocked": False, "record": rec})
    r = scholar.resolve({"title": "Some workshop paper", "url": "https://w.org/p"})
    assert r["status"] == "verified" and r["record"]["source"] == "landing_page"
    assert any("not_in_scholarly_databases" in f for f in r["flags"])


def test_landing_page_other_title_stays_unverified(monkeypatch):
    _no_db(monkeypatch)
    rec = {"source": "landing_page", "title": "A Completely Different Paper", "authors": []}
    monkeypatch.setattr(scholar, "landing_page", lambda url: {"blocked": False, "record": rec})
    r = scholar.resolve({"title": "Fabricated Title Here", "url": "https://w.org/p"})
    assert r["status"] == "unverified"
    assert any("landing_page_shows_other_title" in f for f in r["flags"])
    assert "no_database_match" in r["flags"]


def test_resolve_tolerates_null_fields(monkeypatch):
    monkeypatch.setattr(scholar, "openalex_search_title", lambda *a, **k: [])
    monkeypatch.setattr(scholar, "arxiv_search_title", lambda *a, **k: [])
    r = scholar.resolve({"title": "X Y Z paper", "url": None, "doi": None, "arxiv_id": None, "authors": None})
    assert r["status"] == "unverified"


def test_lookup_error_is_not_fabrication(monkeypatch):
    def boom(c):
        raise RuntimeError("429 from arXiv")
    monkeypatch.setattr(scholar, "resolve", boom)
    out = validate.validate_papers([{"title": "Real Paper", "url": None}])
    assert out["papers"] == [] and out["unverified"][0]["flags"][0].startswith("metadata_lookup_error")


def test_arxiv_by_id_falls_back_to_openalex(monkeypatch):
    import httpx
    def rate_limited(params):
        raise httpx.HTTPStatusError("429", request=httpx.Request("GET", "https://export.arxiv.org"), response=httpx.Response(429))
    monkeypatch.setattr(scholar, "_arxiv_entries", rate_limited)
    monkeypatch.setattr(scholar, "openalex_by_doi", lambda d: {"source": "openalex", "title": "T", "doi": d, "url": None})
    rec = scholar.arxiv_by_id("1605.02026")
    assert rec["arxiv_id"] == "1605.02026" and rec["doi"] == "10.48550/arxiv.1605.02026"
