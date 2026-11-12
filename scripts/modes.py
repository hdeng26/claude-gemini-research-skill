"""read / verify / compare: Gemini reads full texts we fetched; every quote is checked against them."""
from __future__ import annotations

import json
from pathlib import Path

import common
import fulltext
import gemini_client
import scholar
import validate

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

_ITEM = {
    "type": "object",
    "properties": {
        "dimension": {"type": "string"},
        "kind": {"type": "string", "enum": ["stated", "inferred"]},
        "claim": {"type": "string"},
        "quote": {"type": "string"},
        "location": {"type": "string"},
    },
    "required": ["dimension", "claim", "quote", "location"],
}
READ_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "items": {"type": "array", "items": _ITEM},
        "not_stated": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["items"],
}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "contradicted", "conflicting", "insufficient"]},
        "reasoning": {"type": "string"},
        "evidence": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
                "stance": {"type": "string", "enum": ["supports", "contradicts", "context"]},
                "quote": {"type": "string"},
                "location": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["paper_id", "stance", "quote"],
        }},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "evidence"],
}
VERIFY_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "arxiv_id": {"type": ["string", "null"]},
            "doi": {"type": ["string", "null"]},
            "url": {"type": ["string", "null"]},
            "why": {"type": "string"},
        },
        "required": ["title"],
    }}},
    "required": ["candidates"],
}
COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
                "dimension": {"type": "string"},
                "value": {"type": "string"},
                "quote": {"type": "string"},
                "location": {"type": "string"},
            },
            "required": ["paper_id", "dimension", "value", "quote"],
        }},
        "differences": {"type": "array", "items": {
            "type": "object",
            "properties": {"statement": {"type": "string"},
                           "paper_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["statement"],
        }},
        "summary": {"type": "string"},
    },
    "required": ["cells"],
}
DEFAULT_DIMS = ["core method / update rule", "uses backpropagation or not (and what replaces it)",
                "datasets and architectures (depth, width)", "main reported result",
                "stated limitations"]


def _prompt(name, **kw):
    t = (PROMPTS / f"{name}.md").read_text()
    for k, v in kw.items():
        t = t.replace("{{" + k + "}}", str(v))
    return t


def _paper_block(pid: str, ft: dict) -> str:
    rec = ft["record"] or {}
    return (f'<paper id="{pid}">\nTITLE: {rec.get("title")}\n'
            f'AUTHORS: {", ".join((rec.get("authors") or [])[:8])}\nYEAR: {rec.get("year")}\n\n'
            f'{ft["text"]}\n</paper>\n\n')


def _paper_meta(pid: str, ft: dict) -> dict:
    rec = ft["record"] or {}
    return {"paper_id": pid, "title": rec.get("title"), "authors": rec.get("authors") or [],
            "year": rec.get("year"), "doi": rec.get("doi"), "arxiv_id": rec.get("arxiv_id"),
            "url": rec.get("url"), "metadata_source": rec.get("source"),
            "text_source": ft["text_source"], "text_url": ft["text_url"], "chars": ft["chars"],
            "fetch_flags": ft["flags"]}


def _load(idents: list[str]) -> tuple[dict, dict, list]:
    """Fetch full texts. Returns (pid->fulltext, pid->norm texts, failures)."""
    fts, texts, failed = {}, {}, []
    for ident in idents:
        try:
            ft = fulltext.get(ident)
        except (fulltext.FullTextError, Exception) as e:  # network errors too
            failed.append({"ident": ident, "error": f"{type(e).__name__}: {e}"})
            continue
        pid = f"P{len(fts) + 1}"
        fts[pid] = ft
        texts[pid] = {"text": ft["text"], "norm": validate.norm_text(ft["text"])}
    return fts, texts, failed


def _call(prompt, a, label, schema, run, n, tools=()):
    return gemini_client.interact(
        prompt, effort=a.effort, label=label, schema=schema, tools=tools,
        allow_over_daily=a.allow_over_daily, archive=run / f"raw_{n}.json",
        reserve_usd=common.estimate_usd(len(prompt)), timeout=300)


def _finish(run: Path, result: dict, report: str) -> int:
    result["budget_after"] = common.spend_summary()
    common.write_json(run / "result.json", result)
    (run / "report.md").write_text(report)
    print(report)
    return 0


def _head(r: dict, title: str) -> list[str]:
    b = common.spend_summary()
    return [f"# {title}", "", f"effort={r['effort']}  cost=${r['cost_usd']:.4f}  month=${b['month_usd']:.2f}",
            f"run: {r['run_dir']}", ""]


def _item_line(it: dict) -> str:
    s = f"- [{it.get('location') or '?'}] {it.get('claim') or it.get('value') or it.get('note') or ''}"
    if it.get("kind") == "inferred":
        s += "  (inferred by Gemini)"
    s += f"\n  > \"{it.get('quote')}\"  ({it['quote_status']})"
    if it.get("location_warning"):
        s += f"\n  WARNING {it['location_warning']}"
    return s


# ---------------------------------------------------------------- read ------
def _read_cache_path(source: str, effort: str):
    import hashlib
    try:
        ident = fulltext.classify(source)
    except fulltext.FullTextError:
        ident = source
    key = json.dumps([ident, effort], sort_keys=True)
    d = common.CACHE / "read"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")


def do_read(source: str, effort: str, focus: str | None = None, allow_over_daily: bool = False,
            run: Path | None = None, use_cache: bool = True) -> dict:
    """Read one paper. Without a focus question the result is cached per (paper, effort)."""
    cp = _read_cache_path(source, effort) if (use_cache and not focus) else None
    if cp and cp.exists():
        r = json.loads(cp.read_text())
        r.update(cached=True, cost_usd=0.0)
        return r
    fts, texts, failed = _load([source])
    if not fts:
        raise fulltext.FullTextError(failed[0]["error"])
    ft = fts["P1"]
    run = run or common.new_run_dir("read", (ft["record"] or {}).get("title") or source)
    run.mkdir(parents=True, exist_ok=True)
    fb = (f"- focus: answer this question about the paper: {focus}\n" if focus else "")
    prompt = _paper_block("P1", ft) + _prompt("read", focus_block=fb)
    out = gemini_client.interact(prompt, effort=effort, label="read", schema=READ_SCHEMA, tools=(),
                                 allow_over_daily=allow_over_daily, archive=run / "raw_1.json",
                                 reserve_usd=common.estimate_usd(len(prompt)), timeout=300)
    data = out["json"] or {}
    ok, bad = validate.check_items(data.get("items") or [], texts)
    meta = _paper_meta("P1", ft)
    meta["reading_level"] = "full_text_verified" if ok else "full_text_unverified"
    r = {"mode": "read", "source": source, "focus": focus, "effort": effort, "run_dir": str(run),
         "cost_usd": out["cost_usd"], "paper": meta, "summary_model_judgment": data.get("summary"),
         "items": ok, "discarded_items": bad, "not_stated": data.get("not_stated") or []}
    L = _head(r, f"read: {meta['title']}")
    L.append(f"{meta['title']} ({meta['year']}) {', '.join(meta['authors'][:3])}"
             f"{' et al' if len(meta['authors']) > 3 else ''}  doi={meta['doi']} arXiv={meta['arxiv_id']}")
    L.append(f"text: {meta['text_source']} {meta['text_url']} ({meta['chars']} chars)  "
             f"reading_level={meta['reading_level']}")
    for f in meta["fetch_flags"]:
        L.append(f"WARNING {f}")
    L.append(f"quotes verified: {len(ok)}/{len(ok) + len(bad)}")
    if r["summary_model_judgment"]:
        L += ["", "## Summary (Gemini, unchecked)", r["summary_model_judgment"]]
    dims = []
    for it in ok:
        if it["dimension"] not in dims:
            dims.append(it["dimension"])
    for d in dims:
        L += ["", f"## {d}"] + [_item_line(it) for it in ok if it["dimension"] == d]
    if r["not_stated"]:
        L += ["", "## Not stated in the paper (per Gemini)"] + [f"- {x}" for x in r["not_stated"]]
    if bad:
        L += ["", f"## Discarded ({len(bad)}): quote not found in the text — do not use"]
        L += [f"- [{it.get('dimension')}] {it.get('claim')} ({it['quote_status']})" for it in bad]
    r["report"] = "\n".join(L)
    r["budget_after"] = common.spend_summary()
    common.write_json(run / "result.json", r)
    (run / "report.md").write_text(r["report"])
    if cp and ok:
        cp.write_text(json.dumps(r, ensure_ascii=False))
    return r


def cmd_read(a) -> int:
    r = do_read(a.source, a.effort, a.focus, a.allow_over_daily)
    if r.get("cached"):
        print(f"(cached result from {r['run_dir']}; no API call)\n")
    print(r["report"])
    return 0


# -------------------------------------------------------------- verify ------
def cmd_verify(a) -> int:
    run = common.new_run_dir("verify", a.claim)
    cost, n, search_info = 0.0, 1, None
    sources = list(a.source or [])
    if not sources:
        # Find sources first (Google Search), then resolve them to fetchable ids.
        s_out = gemini_client.interact(
            _prompt("verify_search", claim=a.claim, max_sources=a.max_sources),
            effort="low", label="verify-search", schema=VERIFY_SEARCH_SCHEMA,
            allow_over_daily=a.allow_over_daily, archive=run / "raw_search.json")
        cost += s_out["cost_usd"]
        cands = (s_out["json"] or {}).get("candidates") or []
        resolved, unresolved = [], []
        for c in cands[: a.max_sources]:
            res = scholar.resolve(c)
            rec = res["record"]
            ident = rec and (rec.get("arxiv_id") or rec.get("doi"))
            if ident or (res["status"] == "verified" and rec.get("url")):
                resolved.append({"ident": ident or rec["url"], "title": rec["title"], "why": c.get("why"),
                                 "flags": res["flags"]})
            else:
                unresolved.append({"claimed": c, "flags": res["flags"]})
        sources = [x["ident"] for x in resolved]
        search_info = {"queries": s_out["search_queries"], "resolved": resolved, "unresolved": unresolved,
                       "grounding_flags": validate.grounding_flags(s_out)}
    fts, texts, failed = _load(sources)
    r = {"mode": "verify", "claim": a.claim, "effort": a.effort, "run_dir": str(run), "search": search_info,
         "sources": [_paper_meta(p, ft) for p, ft in fts.items()], "fetch_failed": failed}
    if not fts:
        r.update(cost_usd=round(cost, 5), verdict="insufficient", model_verdict=None,
                 verdict_note="no source full text could be obtained", evidence=[], discarded_evidence=[],
                 caveats=[])
    else:
        prompt = "".join(_paper_block(p, ft) for p, ft in fts.items()) + _prompt("verify", claim=a.claim)
        out = _call(prompt, a, "verify", VERIFY_SCHEMA, run, n)
        cost += out["cost_usd"]
        data = out["json"] or {}
        ok, bad = validate.check_items(data.get("evidence") or [], texts)
        verdict, note = validate.final_verdict(data.get("verdict"), ok)
        r.update(cost_usd=round(cost, 5), verdict=verdict, model_verdict=data.get("verdict"),
                 verdict_note=note, reasoning_model_judgment=data.get("reasoning"),
                 evidence=ok, discarded_evidence=bad, caveats=data.get("caveats") or [])
    L = _head(r, f"verify: {a.claim}")
    L.append(f"VERDICT: {r['verdict']}" + (f"   (Gemini said {r['model_verdict']}; {r['verdict_note']})"
                                           if r.get("verdict_note") else ""))
    if search_info:
        L.append("Google queries issued: " + ("; ".join(search_info["queries"]) or "NONE"))
        for f in search_info["grounding_flags"]:
            L.append(f"WARNING {f}")
        for u in search_info["unresolved"]:
            L.append(f"- source Gemini named but not verifiable: {u['claimed'].get('title')} {u['flags']}")
    L += ["", "## Sources read in full"]
    for s in r["sources"]:
        L.append(f"- {s['paper_id']}: {s['title']} ({s['year']}) via {s['text_source']} {s['text_url']}"
                 + (f"  WARNING {s['fetch_flags']}" if s["fetch_flags"] else ""))
    for f in failed:
        L.append(f"- NOT READ: {f['ident']}: {f['error']}")
    for stance in ("supports", "contradicts", "context"):
        ev = [e for e in r["evidence"] if e.get("stance") == stance]
        if ev:
            L += ["", f"## Evidence: {stance}"] + [
                f"- {e['paper_id']} [{e.get('location') or '?'}] {e.get('note') or ''}\n  > \"{e['quote']}\"  ({e['quote_status']})"
                + (f"\n  WARNING {e['location_warning']}" if e.get("location_warning") else "") for e in ev]
    if r.get("caveats"):
        L += ["", "## Caveats (Gemini)"] + [f"- {c}" for c in r["caveats"]]
    if r.get("reasoning_model_judgment"):
        L += ["", "## Reasoning (Gemini, unchecked)", r["reasoning_model_judgment"]]
    if r["discarded_evidence"]:
        L += ["", f"## Discarded evidence ({len(r['discarded_evidence'])}): quote not found — ignore"]
        L += [f"- {e.get('paper_id')} {e.get('stance')}: {str(e.get('quote'))[:120]}" for e in r["discarded_evidence"]]
    L += ["", "A verdict covers only the sources read above. 'insufficient' is not evidence against the claim."]
    return _finish(run, r, "\n".join(L))


# ------------------------------------------------------------- compare ------
def cmd_compare(a) -> int:
    if not 2 <= len(a.sources) <= 6:
        raise SystemExit("compare needs 2-6 sources")
    fts, texts, failed = _load(a.sources)
    if len(fts) < 2:
        raise fulltext.FullTextError(f"need at least 2 readable papers; failures: {failed}")
    dims = a.dims or DEFAULT_DIMS
    run = common.new_run_dir("compare", " vs ".join(((ft["record"] or {}).get("title") or "x")[:20] for ft in fts.values()))
    prompt = ("".join(_paper_block(p, ft) for p, ft in fts.items())
              + _prompt("compare", n=len(fts), dims="\n".join(f"- {d}" for d in dims)))
    out = _call(prompt, a, "compare", COMPARE_SCHEMA, run, 1)
    data = out["json"] or {}
    cells = data.get("cells") or []
    not_stated = [c for c in cells if (c.get("value") or "").strip().lower().startswith("not stated")]
    ok, bad = validate.check_items([c for c in cells if c not in not_stated], texts)
    r = {"mode": "compare", "effort": a.effort, "run_dir": str(run), "cost_usd": out["cost_usd"],
         "dimensions": dims, "papers": [_paper_meta(p, ft) for p, ft in fts.items()], "fetch_failed": failed,
         "cells": ok, "not_stated": not_stated, "discarded_cells": bad,
         "differences_model_judgment": data.get("differences") or [],
         "summary_model_judgment": data.get("summary")}
    L = _head(r, "compare")
    for p in r["papers"]:
        L.append(f"- {p['paper_id']}: {p['title']} ({p['year']}) via {p['text_source']}"
                 + (f"  WARNING {p['fetch_flags']}" if p["fetch_flags"] else ""))
    for f in failed:
        L.append(f"- NOT READ: {f['ident']}: {f['error']}")
    L.append(f"cells verified: {len(ok)}/{len(cells)} ({len(not_stated)} 'not stated')")
    for d in dims:
        L += ["", f"## {d}"]
        for p in r["papers"]:
            pid = p["paper_id"]
            got = [c for c in ok if c["paper_id"] == pid and c.get("dimension") == d]
            ns = [c for c in not_stated if c.get("paper_id") == pid and c.get("dimension") == d]
            if got:
                for c in got:
                    L.append(f"- {pid} [{c.get('location') or '?'}] {c['value']}\n  > \"{c['quote']}\"  ({c['quote_status']})")
            elif ns:
                L.append(f"- {pid}: not stated")
            else:
                L.append(f"- {pid}: (no verified cell)")
    if r["differences_model_judgment"]:
        L += ["", "## Differences (Gemini's judgment, built on the cells)"]
        L += [f"- {d.get('statement')} [{', '.join(d.get('paper_ids') or [])}]" for d in r["differences_model_judgment"]]
    if r["discarded_cells"]:
        L += ["", f"## Discarded cells ({len(bad)}): quote not found — ignore"]
        L += [f"- {c.get('paper_id')} / {c.get('dimension')}: {c.get('value')}" for c in bad]
    return _finish(run, r, "\n".join(L))
