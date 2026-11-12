import json

import pytest

import fulltext
import modes
import research
import scholar
import validate
from conftest import raw_response
from test_core import _queue

TEXT_A = ("## 1 Introduction\nWe propose a layerwise training rule that never propagates errors backwards "
          "through the network. Instead each layer solves a regularised least squares problem in closed "
          "form using the Moore-Penrose pseudoinverse of its input activations.\n## 4 Experiments\n"
          "Table 2 reports test accuracy on MNIST and CIFAR-10 for networks with up to five hidden layers. "
          "Accuracy drops sharply beyond three layers, which we leave to future work. " + "filler text. " * 300)
TEXT_B = ("## 2 Method\nOur method trains deep networks with standard backpropagation but replaces the "
          "optimizer with a second-order least squares step applied after every minibatch. " + "more text. " * 300)


def _ft(text, title):
    return {"record": {"source": "arxiv", "title": title, "authors": ["A Author"], "year": 2024,
                       "doi": None, "arxiv_id": "2401.00001", "url": "u"},
            "text": text, "text_source": "arxiv_html", "text_url": "https://arxiv.org/html/x",
            "truncated": False, "chars": len(text), "flags": []}


@pytest.fixture
def papers(monkeypatch):
    db = {"A": _ft(TEXT_A, "Paper A"), "B": _ft(TEXT_B, "Paper B")}

    def get(ident):
        if ident not in db:
            raise fulltext.FullTextError(f"{ident}: no open-access full text found (paywalled?)")
        return db[ident]
    monkeypatch.setattr(fulltext, "get", get)
    return db


REAL_Q = "each layer solves a regularised least squares problem in closed form using the Moore-Penrose pseudoinverse"
FAKE_Q = "we prove global convergence of the pseudoinverse rule for networks of arbitrary depth"
PARAPHRASE = "every layer computes a regularized least-squares solution analytically via the pseudoinverse"


# ---------- quote matching ------------------------------------------------
def test_match_quote_rules():
    n = validate.norm_text(TEXT_A)
    assert validate.match_quote(REAL_Q, n)["status"] == "exact"
    assert validate.match_quote(FAKE_Q, n)["status"] == "not_found"
    assert validate.match_quote(PARAPHRASE, n)["status"] == "not_found"
    assert validate.match_quote("least squares", n)["status"] == "too_short"
    # PDF artefacts: ligature + end-of-line hyphenation still match
    pdf = "the update to b and subse-\nquent change in normalization led to no change in the ﬁnal output"
    assert validate.match_quote("the update to b and subsequent change in normalization led to no change in the final output",
                                validate.norm_text(pdf))["status"] == "exact"
    # one-character slip -> fuzzy, not rejected
    assert validate.match_quote(REAL_Q.replace("regularised", "regularized"), n)["status"] == "fuzzy"


def test_location_warning():
    assert validate.check_location("Table 2", TEXT_A) is None
    assert "table 7" in validate.check_location("Table 7", TEXT_A)


@pytest.mark.parametrize("model,ev,expect", [
    ("supported", [{"stance": "supports"}], "supported"),
    ("supported", [], "insufficient"),
    ("supported", [{"stance": "supports"}, {"stance": "contradicts"}], "conflicting"),
    ("contradicted", [{"stance": "supports"}], "insufficient"),
    ("conflicting", [{"stance": "supports"}], "insufficient"),
    ("insufficient", [{"stance": "supports"}], "insufficient"),  # never upgraded
    ("definitely true", [{"stance": "supports"}], "insufficient"),
])
def test_final_verdict(model, ev, expect):
    assert validate.final_verdict(model, ev)[0] == expect


# ---------- read ----------------------------------------------------------
def test_read_keeps_only_quote_verified_items(monkeypatch, papers, isolated_home, capsys):
    calls = _queue(monkeypatch, [raw_response(json.dumps({"summary": "s", "items": [
        {"dimension": "optimization", "kind": "stated", "claim": "closed-form LS per layer", "quote": REAL_Q,
         "location": "Section 1"},
        {"dimension": "evidence", "kind": "stated", "claim": "convergence proof", "quote": FAKE_Q, "location": "Theorem 1"},
        {"dimension": "method", "kind": "stated", "claim": "paraphrased", "quote": PARAPHRASE, "location": "Section 1"},
        {"dimension": "evidence", "kind": "stated", "claim": "depth limit",
         "quote": "Accuracy drops sharply beyond three layers, which we leave to future work", "location": "Table 9"},
    ], "not_stated": ["training time"]}), queries=())])
    assert research.main(["read", "A"]) == 0
    assert "tools" not in calls[0]                      # no search for reading
    assert '<paper id="P1">' in calls[0]["input"] and "Moore-Penrose" in calls[0]["input"]
    assert calls[0]["generation_config"]["thinking_level"] == "medium"
    r = json.loads(next((isolated_home / "runs").iterdir()).joinpath("result.json").read_text())
    assert [i["claim"] for i in r["items"]] == ["closed-form LS per layer", "depth limit"]
    assert {i["claim"] for i in r["discarded_items"]} == {"convergence proof", "paraphrased"}
    assert r["paper"]["reading_level"] == "full_text_verified"
    assert "table 9" in r["items"][1]["location_warning"]
    out = capsys.readouterr().out
    assert "quotes verified: 2/4" in out and "Discarded (2)" in out


def test_read_no_fulltext_exit_4_and_no_api_call(monkeypatch, papers):
    calls = _queue(monkeypatch, [])
    assert research.main(["read", "PAYWALLED"]) == 4
    assert calls == []


def test_read_all_quotes_fake_means_unverified_level(monkeypatch, papers, isolated_home):
    _queue(monkeypatch, [raw_response(json.dumps({"items": [
        {"dimension": "method", "claim": "x", "quote": FAKE_Q, "location": "S1"}]}), queries=())])
    research.main(["read", "A"])
    r = json.loads(next((isolated_home / "runs").iterdir()).joinpath("result.json").read_text())
    assert r["paper"]["reading_level"] == "full_text_unverified" and r["items"] == []


# ---------- verify --------------------------------------------------------
def _verify_reply(verdict, evidence):
    return raw_response(json.dumps({"verdict": verdict, "reasoning": "r", "evidence": evidence, "caveats": []}),
                        queries=())


def _result(home):
    return json.loads(next((home / "runs").iterdir()).joinpath("result.json").read_text())


def test_verify_supported_with_fabricated_quote_is_downgraded(monkeypatch, papers, isolated_home):
    _queue(monkeypatch, [_verify_reply("supported", [{"paper_id": "P1", "stance": "supports", "quote": FAKE_Q}])])
    assert research.main(["verify", "Paper A proves convergence for any depth", "--source", "A"]) == 0
    r = _result(isolated_home)
    assert r["verdict"] == "insufficient" and r["model_verdict"] == "supported"
    assert "downgraded" in r["verdict_note"] and r["discarded_evidence"]


def test_verify_contradicted_with_real_quote(monkeypatch, papers, isolated_home):
    _queue(monkeypatch, [_verify_reply("contradicted", [{"paper_id": "P1", "stance": "contradicts",
        "quote": "Accuracy drops sharply beyond three layers, which we leave to future work", "location": "Section 4"}])])
    research.main(["verify", "Paper A works well at any depth", "--source", "A"])
    assert _result(isolated_home)["verdict"] == "contradicted"


def test_verify_quote_attributed_to_wrong_paper_is_discarded(monkeypatch, papers, isolated_home):
    _queue(monkeypatch, [_verify_reply("supported", [{"paper_id": "P2", "stance": "supports", "quote": REAL_Q}])])
    research.main(["verify", "claim", "--source", "A", "--source", "B"])
    r = _result(isolated_home)
    assert r["verdict"] == "insufficient" and r["discarded_evidence"][0]["paper_id"] == "P2"


def test_verify_without_sources_searches_then_reads(monkeypatch, papers, isolated_home):
    monkeypatch.setattr(scholar, "resolve", lambda c: (
        {"status": "verified", "record": {"title": "Paper A", "arxiv_id": "A", "doi": None, "url": "u"}, "flags": []}
        if c["title"] == "Paper A" else {"status": "unverified", "record": None, "flags": ["no_database_match"]}))
    calls = _queue(monkeypatch, [
        raw_response(json.dumps({"candidates": [{"title": "Paper A", "why": "w"}, {"title": "Ghost Paper"}]}),
                     queries=("q1", "q2")),
        _verify_reply("supported", [{"paper_id": "P1", "stance": "supports", "quote": REAL_Q}])])
    research.main(["verify", "Paper A uses a closed-form least squares step per layer"])
    assert calls[0]["tools"] == [{"type": "google_search"}, {"type": "url_context"}]
    assert calls[0]["generation_config"]["thinking_level"] == "low"
    r = _result(isolated_home)
    assert r["verdict"] == "supported"
    assert r["search"]["queries"] == ["q1", "q2"]
    assert r["search"]["unresolved"][0]["claimed"]["title"] == "Ghost Paper"


def test_verify_no_readable_source_is_insufficient_without_judge_call(monkeypatch, papers, isolated_home):
    calls = _queue(monkeypatch, [])
    assert research.main(["verify", "claim", "--source", "PAYWALLED"]) == 0
    r = _result(isolated_home)
    assert r["verdict"] == "insufficient" and calls == [] and r["fetch_failed"]


# ---------- compare -------------------------------------------------------
def test_compare_checks_each_cell_against_its_own_paper(monkeypatch, papers, isolated_home, capsys):
    _queue(monkeypatch, [raw_response(json.dumps({"cells": [
        {"paper_id": "P1", "dimension": "bp", "value": "no BP", "quote": "a layerwise training rule that never propagates errors backwards through the network"},
        {"paper_id": "P2", "dimension": "bp", "value": "uses BP", "quote": "Our method trains deep networks with standard backpropagation"},
        {"paper_id": "P2", "dimension": "depth", "value": "5 layers", "quote": "networks with up to five hidden layers"},
        {"paper_id": "P1", "dimension": "time", "value": "not stated", "quote": ""},
    ], "differences": [{"statement": "A avoids BP, B does not", "paper_ids": ["P1", "P2"]}], "summary": "s"}),
        queries=())])
    assert research.main(["compare", "A", "B", "--dim", "bp", "--dim", "depth", "--dim", "time"]) == 0
    r = _result(isolated_home)
    assert len(r["cells"]) == 2
    assert r["discarded_cells"][0]["dimension"] == "depth"   # quote is from P1, attributed to P2
    assert r["not_stated"][0]["dimension"] == "time"
    out = capsys.readouterr().out
    assert "P2: (no verified cell)" in out and "P1: not stated" in out


def test_compare_needs_two_readable(monkeypatch, papers):
    _queue(monkeypatch, [])
    assert research.main(["compare", "A", "PAYWALLED"]) == 4
