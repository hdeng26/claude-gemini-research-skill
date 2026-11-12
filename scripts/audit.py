"""audit: check a deep-research report (e.g. exported from the Gemini app) against its own sources.

1. Gemini extracts the reference list and atomic claims; every report_quote and URL is
   checked against the report text, so extraction errors cannot slip in.
2. References: bibliography checked (arXiv/Crossref/OpenAlex); source text fetched.
3. Claims: for each, the most relevant passages of its cited sources are found locally
   (keyword scoring, no LLM), Gemini judges claim vs passages, quotes are checked against
   the full source text. Only verified quotes can make a claim "supported"/"contradicted".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import common
import fulltext
import gemini_client
import scholar
import validate

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
KIND_PRIORITY = {"number": 0, "novelty": 1, "comparison": 2, "method": 3, "other": 4}
STATUS_LABEL = {
    "supported": "Supported by the cited source",
    "contradicted": "Contradicted by the cited source",
    "not_found": "Not found in the cited source passages",
    "judgment_unverified": "Judgment without a matching quote (treat as unchecked)",
    "source_unavailable": "Cited source unavailable",
    "uncited": "No citation",
    "not_checked_budget": "Not checked (cost cap reached)",
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "references": {"type": "array", "items": {
            "type": "object",
            "properties": {"ref_id": {"type": "string"}, "title": {"type": "string"},
                           "url": {"type": ["string", "null"]}, "report_quote": {"type": "string"}},
            "required": ["ref_id", "title"]}},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"claim_id": {"type": "string"}, "claim": {"type": "string"},
                           "report_quote": {"type": "string"},
                           "ref_ids": {"type": "array", "items": {"type": "string"}},
                           "kind": {"type": "string", "enum": list(KIND_PRIORITY)}},
            "required": ["claim", "report_quote", "ref_ids"]}},
    },
    "required": ["references", "claims"],
}
CHECK_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string"},
            "judgment": {"type": "string", "enum": ["supported", "contradicted", "not_found"]},
            "evidence": {"type": "array", "items": {
                "type": "object",
                "properties": {"ref_id": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["ref_id", "quote"]}},
            "note": {"type": "string"}},
        "required": ["claim_id", "judgment"]}}},
    "required": ["results"],
}

_STOP = set("""a an the of and or to in on for with by from as at is are was were be been being this that these
those it its into than then which who whom whose what when where how why not no can could may might will would
shall should do does did has have had their there they them we our us you your he she his her also such using use
used based more most less very only both each other via per""".split())


def _prompt(name, **kw):
    t = (PROMPTS / f"{name}.md").read_text()
    for k, v in kw.items():
        t = t.replace("{{" + k + "}}", str(v))
    return t


def load_report(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        raise fulltext.FullTextError(f"report file not found: {p}")
    data = p.read_bytes()
    if data[:5] == b"%PDF-":
        return fulltext.pdf_to_text(data)
    text = data.decode("utf-8", errors="replace")
    if p.suffix.lower() in (".html", ".htm"):
        # keep hrefs visible so reference URLs survive
        text = re.sub(r'<a\s[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r"\2 (\1)", text, flags=re.S | re.I)
        return fulltext.html_to_text(text)
    return text


def _words(s: str) -> list[str]:
    return [w for w in validate.norm_text(s).split() if w not in _STOP and (len(w) > 2 or w.isdigit())]


def top_passages(claim: str, text: str, k: int = 3, size: int = 1400) -> list[str]:
    """Pick the k windows of `text` that share the most content words (numbers count triple)."""
    cw = set(_words(claim))
    if not cw or not text:
        return []
    paras = []
    for p in re.split(r"\n\s*\n|\n(?=## )", text):
        if not p.strip():
            continue
        while len(p) > size:  # PDF text often has no blank lines: cut long blocks at whitespace
            cut = p.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            paras.append(p[:cut])
            p = p[cut:]
        paras.append(p)
    windows, cur = [], ""
    for p in paras:  # pack paragraphs into ~size windows
        if len(cur) + len(p) > size and cur:
            windows.append(cur)
            cur = cur[-200:]  # small overlap
        cur += "\n" + p
    if cur.strip():
        windows.append(cur)
    out = []
    for w in windows:
        ws = set(_words(w))
        score = sum(3 if t.isdigit() else 1 for t in cw & ws)
        if score:
            out.append((score, w))
    out.sort(key=lambda x: -x[0])
    return [w.strip() for _, w in out[:k]]


def _check_report_quotes(items, report_norm):
    ok, bad = [], []
    for it in items:
        m = validate.match_quote(it.get("report_quote") or "", report_norm)
        (ok if m["status"] in ("exact", "fuzzy") else bad).append(it)
    return ok, bad


def _resolve_ref(ref: dict, report_text: str) -> dict:
    out = {"ref_id": ref["ref_id"], "title": ref.get("title"), "url": ref.get("url"), "flags": []}
    if out["url"] and out["url"] not in report_text:
        out["flags"].append("url_not_in_report (extraction error); ignored")
        out["url"] = None
    res = scholar.resolve({"title": ref.get("title") or "", "url": out["url"] or ""})
    out["bib_status"] = "paper_verified" if res["status"] == "verified" else "not_in_databases"
    out["bib_flags"] = res["flags"]
    rec = res["record"] or {}
    if rec:
        out["record"] = {k: rec.get(k) for k in ("title", "authors", "year", "doi", "arxiv_id", "venue", "source")}
    tries = [x for x in (rec.get("arxiv_id"), rec.get("doi"), out["url"]) if x]
    out["text_status"], out["text"] = "unavailable", None
    errors = []
    for t in tries:
        try:
            ft = fulltext.get(t)
        except Exception as e:
            errors.append(f"{t}: {type(e).__name__}: {str(e)[:120]}")
            continue
        out.update(text=ft["text"], text_status="ok", text_source=ft["text_source"], text_url=ft["text_url"])
        out["flags"] += ft["flags"]
        if out["bib_status"] != "paper_verified":
            page_title = (ft["record"] or {}).get("title")
            if page_title and scholar.title_sim(page_title, ref.get("title") or "") < 0.6:
                out["flags"].append(f"page_title_differs: {page_title[:80]}")
            out["bib_status"] = "web_page"
        break
    if out["text_status"] != "ok":
        out["text_errors"] = errors or ["no URL and no identifier to fetch"]
    return out


def run_audit(a) -> dict:
    report_text = load_report(a.report)
    report_norm = validate.norm_text(report_text)
    run = common.new_run_dir("audit", Path(a.report).stem)
    common.set_run_cap(a.max_cost)
    cost = 0.0
    try:
        ex = gemini_client.interact(
            f"<report>\n{report_text}\n</report>\n\n" + _prompt("audit_extract", max_claims=a.max_claims),
            effort="low", label="audit-extract", schema=EXTRACT_SCHEMA, tools=(),
            allow_over_daily=a.allow_over_daily, archive=run / "raw_extract.json",
            reserve_usd=common.estimate_usd(len(report_text), 6000), timeout=300)
        cost += ex["cost_usd"]
        data = ex["json"] or {}
        claims, bad_claims = _check_report_quotes(data.get("claims") or [], report_norm)
        for i, c in enumerate(claims, 1):
            c["claim_id"] = f"C{i}"
            c["ref_ids"] = [str(x).strip("[] ") for x in c.get("ref_ids") or []]
        claims.sort(key=lambda c: KIND_PRIORITY.get(c.get("kind"), 4))
        claims = claims[: a.max_claims]

        refs = {}
        for r in data.get("references") or []:
            rid = str(r.get("ref_id", "")).strip("[]. ")
            if rid and rid not in refs:
                refs[rid] = _resolve_ref(dict(r, ref_id=rid), report_text)
        texts = {rid: {"text": r["text"], "norm": validate.norm_text(r["text"])}
                 for rid, r in refs.items() if r["text"]}

        results = {}
        todo = []
        for c in claims:
            cited = [rid for rid in c["ref_ids"] if rid in refs]
            if not c["ref_ids"]:
                results[c["claim_id"]] = {"status": "uncited"}
            elif not any(rid in texts for rid in cited):
                results[c["claim_id"]] = {"status": "source_unavailable",
                                          "note": "cited: " + ", ".join(c["ref_ids"])}
            else:
                todo.append(c)
        for i in range(0, len(todo), a.batch):
            batch = todo[i:i + a.batch]
            blocks, lines = {}, []
            for c in batch:
                for rid in c["ref_ids"]:
                    if rid in texts:
                        for p in top_passages(c["claim"] + " " + c["report_quote"], texts[rid]["text"]):
                            blocks.setdefault(rid, [])
                            if p not in blocks[rid]:
                                blocks[rid].append(p)
                lines.append(f'- {c["claim_id"]} (cites {", ".join(c["ref_ids"])}): {c["claim"]}')
            src = "".join(f'<source ref_id="{rid}" title="{refs[rid].get("title")}">\n' + "\n...\n".join(ps)
                          + "\n</source>\n\n" for rid, ps in blocks.items())
            prompt = src + _prompt("audit_check", claims="\n".join(lines))
            try:
                out = gemini_client.interact(prompt, effort=a.effort, label="audit-check", schema=CHECK_SCHEMA,
                                             tools=(), allow_over_daily=a.allow_over_daily,
                                             archive=run / f"raw_check_{i // a.batch + 1}.json",
                                             reserve_usd=common.estimate_usd(len(prompt), 4000))
            except common.BudgetError:  # run cap or daily soft cap
                for c in todo[i:]:
                    results[c["claim_id"]] = {"status": "not_checked_budget"}
                break
            cost += out["cost_usd"]
            by_id = {r.get("claim_id"): r for r in (out["json"] or {}).get("results") or []}
            for c in batch:
                r = by_id.get(c["claim_id"]) or {"judgment": "not_found", "note": "model returned no result"}
                ev = [dict(e, paper_id=str(e.get("ref_id", "")).strip("[] ")) for e in r.get("evidence") or []]
                cited_texts = {rid: texts[rid] for rid in c["ref_ids"] if rid in texts}
                ok, badq = validate.check_items(ev, cited_texts) if ev else ([], [])
                j = r.get("judgment")
                if j in ("supported", "contradicted"):
                    status = j if ok else "judgment_unverified"
                else:
                    status = "not_found"
                results[c["claim_id"]] = {"status": status, "model_judgment": j, "note": r.get("note"),
                                          "evidence": ok, "discarded_evidence": badq}
    finally:
        common.set_run_cap(None)

    for c in claims:
        c.update(results.get(c["claim_id"], {"status": "not_checked_budget"}))
    counts = {}
    for c in claims:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    ref_list = [{k: v for k, v in r.items() if k != "text"} for r in refs.values()]
    r = {"mode": "audit", "report": str(Path(a.report).expanduser()), "effort": a.effort, "run_dir": str(run),
         "cost_usd": round(cost, 5), "counts": counts, "claims": claims,
         "extraction_rejected_claims": bad_claims, "references": ref_list}
    r["report_md"] = render(r)
    r["budget_after"] = common.spend_summary()
    common.write_json(run / "result.json", r)
    (run / "report.md").write_text(r["report_md"])
    return r


def render(r: dict) -> str:
    L = [f"# audit: {Path(r['report']).name}", "",
         f"effort={r['effort']}  cost=${r['cost_usd']:.4f}  run: {r['run_dir']}", "",
         "## Summary"]
    for s, label in STATUS_LABEL.items():
        if r["counts"].get(s):
            L.append(f"- {label} ({s}): {r['counts'][s]}")
    if r["extraction_rejected_claims"]:
        L.append(f"- extraction errors dropped (quote not in report): {len(r['extraction_rejected_claims'])}")
    order = ["contradicted", "judgment_unverified", "not_found", "source_unavailable", "uncited",
             "not_checked_budget", "supported"]
    for s in order:
        cs = [c for c in r["claims"] if c["status"] == s]
        if not cs:
            continue
        L += ["", f"## {STATUS_LABEL[s]} ({s}) — {len(cs)}"]
        for c in cs:
            L.append(f"- {c['claim_id']} [{c.get('kind')}] {c['claim']}  (cites {', '.join(c['ref_ids']) or '-'})")
            if c.get("note"):
                L.append(f"  note (Gemini): {c['note']}")
            for e in c.get("evidence") or []:
                L.append(f"  > [{e['paper_id']}] \"{e['quote']}\"  ({e['quote_status']})")
    L += ["", "## References"]
    for ref in r["references"]:
        rec = ref.get("record") or {}
        L.append(f"- [{ref['ref_id']}] {ref.get('title')}  bib={ref['bib_status']}  text={ref['text_status']}"
                 + (f" via {ref.get('text_source')}" if ref.get("text_source") else "")
                 + (f"  db title: {rec.get('title')} ({rec.get('year')})" if rec and ref["bib_status"] == "paper_verified" else ""))
        for f in ref["flags"] + [f for f in ref.get("bib_flags", []) if ref["bib_status"] == "paper_verified"]:
            L.append(f"  WARNING {f}")
        if ref["text_status"] != "ok":
            L.append(f"  not readable: {'; '.join(ref.get('text_errors') or [])[:200]}")
    L += ["", "not_found means the locally selected passages did not settle it; the source may still say it elsewhere.",
          "supported/contradicted require a quote that was found verbatim in the source text."]
    return "\n".join(L)


def cmd_audit(a) -> int:
    r = run_audit(a)
    print(r["report_md"])
    return 0
