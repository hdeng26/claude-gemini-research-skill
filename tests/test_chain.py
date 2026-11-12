import json

import pytest

import audit
import background
import common
import fulltext
import modes
import research
import scholar
import scout
from conftest import raw_response
from test_core import _queue
from test_modes import TEXT_A, _ft

REPORT = """# Deep research: layerwise training

Layerwise closed-form training is an old idea. Each layer solves a regularised least squares problem in closed form using the Moore-Penrose pseudoinverse of its input activations [1].
The method keeps working with more than ten hidden layers without any loss in accuracy [1].
It was the first method ever to train networks without backpropagation [1].
Pseudoinverse methods were also scaled to ImageNet in 2019 [2].
Many researchers find this direction promising for future hardware.

## Works cited
1. Paper A, https://arxiv.org/abs/2401.00001
2. Imaginary Scaling Study, https://example.org/missing
"""


def _write_report(tmp_path):
    p = tmp_path / "dr.md"
    p.write_text(REPORT)
    return p


@pytest.fixture
def sources(monkeypatch):
    def get(ident):
        if ident in ("2401.00001", "https://arxiv.org/abs/2401.00001"):
            return _ft(TEXT_A, "Paper A")
        raise fulltext.FullTextError(f"{ident}: http 404")
    monkeypatch.setattr(fulltext, "get", get)

    def resolve(c):
        if c.get("title") == "Paper A":
            return {"status": "verified", "flags": [], "title_similarity": 1.0, "checked": [],
                    "record": {"source": "arxiv", "title": "Paper A", "authors": [], "year": 2024,
                               "doi": None, "arxiv_id": "2401.00001", "url": "u"}}
        return {"status": "unverified", "record": None, "flags": ["no_database_match"], "checked": []}
    monkeypatch.setattr(scholar, "resolve", resolve)


def _extract_reply():
    return raw_response(json.dumps({
        "references": [
            {"ref_id": "1", "title": "Paper A", "url": "https://arxiv.org/abs/2401.00001", "report_quote": "1. Paper A"},
            {"ref_id": "2", "title": "Imaginary Scaling Study", "url": "https://example.org/missing"},
            {"ref_id": "3", "title": "Hallucinated ref", "url": "https://not-in-report.org/x"},
        ],
        "claims": [
            {"claim": "each layer is solved in closed form with the pseudoinverse", "kind": "method", "ref_ids": ["1"],
             "report_quote": "Each layer solves a regularised least squares problem in closed form using the Moore-Penrose pseudoinverse of its input activations"},
            {"claim": "works beyond ten layers without accuracy loss", "kind": "number", "ref_ids": ["[1]"],
             "report_quote": "The method keeps working with more than ten hidden layers without any loss in accuracy"},
            {"claim": "first method to train without backprop", "kind": "novelty", "ref_ids": ["1"],
             "report_quote": "It was the first method ever to train networks without backpropagation"},
            {"claim": "scaled to ImageNet in 2019", "kind": "number", "ref_ids": ["2"],
             "report_quote": "Pseudoinverse methods were also scaled to ImageNet in 2019"},
            {"claim": "researchers find it promising", "kind": "other", "ref_ids": [],
             "report_quote": "Many researchers find this direction promising for future hardware"},
            {"claim": "made-up extraction", "kind": "number", "ref_ids": ["1"],
             "report_quote": "This sentence does not appear anywhere in the report at all"},
        ]}), queries=())


def _check_reply():
    return raw_response(json.dumps({"results": [
        {"claim_id": "C1", "judgment": "supported", "evidence": [{"ref_id": "1", "quote":
            "each layer solves a regularised least squares problem in closed form using the Moore-Penrose pseudoinverse"}]},
        {"claim_id": "C2", "judgment": "contradicted", "evidence": [{"ref_id": "1", "quote":
            "Accuracy drops sharply beyond three layers, which we leave to future work"}]},
        {"claim_id": "C3", "judgment": "supported", "evidence": [{"ref_id": "1", "quote":
            "this is the first method ever proposed to train networks without backpropagation"}]},
    ]}), queries=())


def test_audit_statuses(monkeypatch, sources, isolated_home, capsys):
    calls = _queue(monkeypatch, [_extract_reply(), _check_reply()])
    rep = _write_report(isolated_home)
    assert research.main(["audit", str(rep)]) == 0
    assert all("tools" not in c for c in calls)
    r = json.loads(next((isolated_home / "runs").iterdir()).joinpath("result.json").read_text())
    st = {c["claim"]: c["status"] for c in r["claims"]}
    assert st["each layer is solved in closed form with the pseudoinverse"] == "supported"
    assert st["works beyond ten layers without accuracy loss"] == "contradicted"
    assert st["first method to train without backprop"] == "judgment_unverified"  # fabricated quote
    assert st["scaled to ImageNet in 2019"] == "source_unavailable"
    assert st["researchers find it promising"] == "uncited"
    assert "made-up extraction" not in st and len(r["extraction_rejected_claims"]) == 1
    refs = {x["ref_id"]: x for x in r["references"]}
    assert refs["1"]["bib_status"] == "paper_verified" and refs["1"]["text_status"] == "ok"
    assert refs["2"]["text_status"] == "unavailable"
    assert any("url_not_in_report" in f for f in refs["3"]["flags"])
    # passages sent for checking come from the cited source, not the whole text
    assert '<source ref_id="1"' in calls[1]["input"] and '<source ref_id="2"' not in calls[1]["input"]
    out = capsys.readouterr().out
    assert "Contradicted by the cited source" in out and "Cited source unavailable" in out


def test_audit_run_cap_marks_rest_not_checked(monkeypatch, sources, isolated_home):
    calls = _queue(monkeypatch, [_extract_reply(), _check_reply()])
    rep = _write_report(isolated_home)
    assert research.main(["audit", str(rep), "--max-cost", "0.06"]) == 0
    assert len(calls) == 1  # check call refused by the run cap
    r = json.loads(next((isolated_home / "runs").iterdir()).joinpath("result.json").read_text())
    assert r["counts"].get("not_checked_budget") == 3
    assert common._RUN["cap"] is None  # cap cleared afterwards


def test_top_passages_bounded_on_text_without_blank_lines():
    text = ("lorem ipsum dolor " * 3000) + " Accuracy drops sharply beyond three layers. " + ("sit amet " * 3000)
    ps = audit.top_passages("accuracy drops beyond three layers", text, k=3, size=1400)
    assert ps and all(len(p) <= 1700 for p in ps)
    assert "three layers" in ps[0]


def test_top_passages_prefers_numbers_and_overlap():
    text = "## A\nunrelated words about cats.\n\n## B\nAccuracy drops sharply beyond three layers on CIFAR-10.\n\n## C\nmore cats"
    ps = audit.top_passages("accuracy beyond three layers", text, k=1, size=40)
    assert "three layers" in ps[0]


# ---------------------------------------------------------------- background
def _paper(i, arxiv=True, abstract="abs"):
    return {"title": f"Paper {i}", "authors": [], "year": 2020 + i, "venue": "V", "doi": None,
            "arxiv_id": f"240{i}.0000{i}" if arxiv else None, "url": None, "abstract": abstract,
            "reading_level": "abstract_verified", "metadata_source": "arxiv", "flags": [], "model_judgment": {}}


@pytest.fixture
def chain(monkeypatch):
    calls = {"scout": [], "read": []}

    def fake_scout(topic, effort, max_papers=10, allow_over_daily=False, run=None, extra="", label="scout"):
        calls["scout"].append({"topic": topic, "effort": effort, "extra": extra})
        n = len(calls["scout"])
        return {"papers": [_paper(1), _paper(n + 1)], "unverified": [], "run_dir": str(run),
                "search_queries_issued": [f"q{n}"], "grounding_flags": []}
    monkeypatch.setattr(scout, "do_scout", fake_scout)

    def fake_read(ident, effort, focus=None, allow_over_daily=False, run=None, use_cache=True):
        calls["read"].append((ident, effort))
        if ident == "2402.00002":
            raise fulltext.FullTextError("paywalled")
        return {"items": [{"dimension": "method", "claim": f"claim of {ident}", "quote": f"quote from {ident} ok",
                           "location": "S1", "quote_status": "exact"}], "discarded_items": [], "run_dir": "x",
                "cost_usd": 0.0}
    monkeypatch.setattr(modes, "do_read", fake_read)
    return calls


def _synth(evidence_ids):
    return raw_response(json.dumps({
        "closest_work": [{"paper": "Paper 1", "similarity": "very", "key_difference": "d", "evidence": evidence_ids[0]}],
        "novelty_threats": [{"statement": "t", "evidence": evidence_ids[1]}],
        "open_gaps": [{"gap": "g", "why_it_looks_open": "w", "evidence": evidence_ids[2]}],
        "unresolved_questions": ["u"], "summary": "s"}), queries=())


def test_background_full_chain(monkeypatch, chain, isolated_home, capsys):
    calls = _queue(monkeypatch, [
        raw_response(json.dumps({"angles": [{"angle": "closest", "search_topic": "t1"},
                                            {"angle": "threats", "search_topic": "t2"}]}), queries=()),
        raw_response(json.dumps({"selected": [{"id": "S3", "reason": "r"}, {"id": "S1", "reason": "r"},
                                              {"id": "S99", "reason": "not in list"}, {"id": "S2"}]}), queries=()),
        _synth([["P1.E1"], ["S99"], ["S3"]]),
    ])
    assert research.main(["background", "closed-form layer training depth", "--angles", "2", "--read", "2"]) == 0
    assert all(c["generation_config"]["thinking_level"] == "high" for c in calls)
    assert [s["effort"] for s in chain["scout"]] == ["high", "high"]
    # Paper 1 found by both angles -> merged, angles=2
    r = json.loads(next((isolated_home / "runs").iterdir()).joinpath("result.json").read_text())
    p1 = next(p for p in r["papers"] if p["title"] == "Paper 1")
    assert p1["angles"] == 2
    # S3 (Paper 2, older) is paywalled -> topped up with the next selected paper
    assert [x["title"] for x in r["read"]] == ["Paper 1", "Paper 3"]
    assert r["read_failed"][0]["title"] == "Paper 2"
    b = r["brief"]
    assert b["closest_work"][0]["support"] == "full_text"
    assert b["novelty_threats"][0]["support"] == "none" and b["novelty_threats"][0]["bad_evidence_ids"] == ["S99"]
    out = capsys.readouterr().out
    assert "NO valid evidence ids" in out


def test_background_reuses_topic_cache(monkeypatch, chain, isolated_home):
    _queue(monkeypatch, [
        raw_response(json.dumps({"angles": [{"angle": "a", "search_topic": "t1"}]}), queries=()),
        raw_response(json.dumps({"selected": [{"id": "S1"}]}), queries=()), _synth([["P1.E1"], [], []]),
        raw_response(json.dumps({"selected": [{"id": "S1"}]}), queries=()), _synth([["P1.E1"], [], []]),
    ])
    research.main(["background", "closed-form layer training depth", "--angles", "1", "--read", "1"])
    research.main(["background", "Closed-Form Layer Training DEPTH", "--angles", "1", "--read", "1"])
    assert len(chain["scout"]) == 2
    assert "on or after" in chain["scout"][1]["extra"]


def test_background_run_cap_stops_but_reports(monkeypatch, chain, isolated_home, capsys):
    _queue(monkeypatch, [raw_response(json.dumps({"angles": [{"angle": "a", "search_topic": "t1"}]}), queries=()),
                         raw_response(json.dumps({"selected": [{"id": "S1"}]}), queries=())])
    assert research.main(["background", "x", "--angles", "1", "--max-cost", "0.24"]) == 0
    out = capsys.readouterr().out
    assert "STOPPED EARLY" in out
    assert common._RUN["cap"] is None


def test_background_survives_one_failed_scout(monkeypatch, chain, isolated_home, capsys):
    orig = scout.do_scout

    def flaky(topic, *a, **k):
        if topic == "t1":
            raise AttributeError("boom")
        return orig(topic, *a, **k)
    monkeypatch.setattr(scout, "do_scout", flaky)
    _queue(monkeypatch, [
        raw_response(json.dumps({"angles": [{"angle": "a", "search_topic": "t1"}, {"angle": "b", "search_topic": "t2"}]}), queries=()),
        raw_response(json.dumps({"selected": [{"id": "S1"}]}), queries=()), _synth([["P1.E1"], [], []])])
    assert research.main(["background", "x", "--angles", "2", "--read", "1"]) == 0
    assert "scout failed: AttributeError" in capsys.readouterr().out


def test_background_daily_cap_keeps_partial_result(monkeypatch, chain, isolated_home, capsys):
    common.record_spend({"label": "x", "cost_usd": 1.78, "search_queries": 0})
    _queue(monkeypatch, [raw_response(json.dumps({"angles": [{"angle": "a", "search_topic": "t1"}]}), queries=()),
                         raw_response(json.dumps({"selected": [{"id": "S1"}]}), queries=())])
    assert research.main(["background", "x", "--angles", "1", "--read", "1"]) == 0
    out = capsys.readouterr().out
    assert "STOPPED EARLY" in out and "daily soft cap" in out and "Paper 1" in out
