"""background: chained scout -> select -> read -> synthesize, all at high thinking level.

Meant to run before planning research work. Caches:
  - paper level: `read` results (no focus) are cached per paper+effort, so a paper is read once.
  - topic level: the merged scout list per topic is kept; a rerun within --reuse-days only adds
    one "recent work" scout and merges, instead of redoing every angle.
A per-run cost cap (--max-cost) stops the chain early; whatever finished is still reported.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common
import gemini_client
import modes
import scholar
import scout

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

ANGLES_SCHEMA = {"type": "object", "properties": {"angles": {"type": "array", "items": {
    "type": "object", "properties": {"angle": {"type": "string"}, "search_topic": {"type": "string"}},
    "required": ["angle", "search_topic"]}}}, "required": ["angles"]}
SELECT_SCHEMA = {"type": "object", "properties": {"selected": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["id"]}}}, "required": ["selected"]}
_STMT = {"type": "array", "items": {"type": "string"}}
SYNTH_SCHEMA = {"type": "object", "properties": {
    "closest_work": {"type": "array", "items": {"type": "object", "properties": {
        "paper": {"type": "string"}, "similarity": {"type": "string"}, "key_difference": {"type": "string"},
        "evidence": _STMT}, "required": ["paper", "similarity", "evidence"]}},
    "novelty_threats": {"type": "array", "items": {"type": "object", "properties": {
        "statement": {"type": "string"}, "evidence": _STMT}, "required": ["statement", "evidence"]}},
    "open_gaps": {"type": "array", "items": {"type": "object", "properties": {
        "gap": {"type": "string"}, "why_it_looks_open": {"type": "string"}, "evidence": _STMT},
        "required": ["gap", "evidence"]}},
    "unresolved_questions": _STMT,
    "summary": {"type": "string"}}, "required": ["closest_work", "novelty_threats", "open_gaps"]}


def _prompt(name, **kw):
    t = (PROMPTS / f"{name}.md").read_text()
    for k, v in kw.items():
        t = t.replace("{{" + k + "}}", str(v))
    return t


def _topic_cache(topic: str) -> Path:
    d = common.CACHE / "background"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha256(scholar.norm_title(topic).encode()).hexdigest()[:24] + ".json")


def _key(p: dict) -> str:
    return (p.get("doi") or (p.get("arxiv_id") and f"arxiv:{p['arxiv_id']}") or scholar.norm_title(p["title"]))


def _ident(p: dict) -> str | None:
    return p.get("arxiv_id") or p.get("doi") or (p.get("url") if (p.get("url") or "").startswith("http") else None)


def merge(lists: list[list[dict]]) -> list[dict]:
    seen: dict[str, dict] = {}
    for lst in lists:
        for p in lst:
            k = _key(p)
            if k in seen:
                seen[k]["angles"] = seen[k].get("angles", 1) + 1
            else:
                seen[k] = dict(p, angles=p.get("angles", 1))
    return sorted(seen.values(), key=lambda p: (-p["angles"], -(p.get("year") or 0)))


def run_background(a) -> dict:
    run = common.new_run_dir("background", a.topic)
    log, cost_before = [], common.spend_summary()["month_usd"]
    common.set_run_cap(a.max_cost)
    stopped = None
    angles, scouts, papers, selected, reads, read_failed, synth = [], [], [], [], [], [], None
    prev = None
    tc = _topic_cache(a.topic)
    if tc.exists() and not a.fresh:
        prev = json.loads(tc.read_text())
        age = (dt.date.today() - dt.date.fromisoformat(prev["date"])).days
        if age > a.reuse_days:
            prev = None
    try:
        # 1-2. discover
        if prev:
            log.append(f"reusing topic cache from {prev['date']}: {len(prev['papers'])} papers; one recent-work scout")
            r = scout.do_scout(a.topic, a.effort, 8, a.allow_over_daily, run=run / "scout_recent",
                               extra=f"Only list papers that appeared on or after {prev['date']}.",
                               label="bg-scout")
            scouts.append({"angle": f"new since {prev['date']}", "n": len(r["papers"]), "run": r["run_dir"],
                           "queries": r["search_queries_issued"], "flags": r["grounding_flags"]})
            papers = merge([prev["papers"], r["papers"]])
        else:
            out = gemini_client.interact(_prompt("background_angles", topic=a.topic, n=a.angles),
                                         effort=a.effort, label="bg-angles", schema=ANGLES_SCHEMA, tools=(),
                                         allow_over_daily=a.allow_over_daily, archive=run / "raw_angles.json")
            angles = ((out["json"] or {}).get("angles") or [])[: a.angles]
            if not angles:
                angles = [{"angle": "direct", "search_topic": a.topic}]
            results = []

            def _scout(i_ang):
                i, ang = i_ang
                try:
                    return i, ang, scout.do_scout(ang["search_topic"], a.effort, 8, a.allow_over_daily,
                                                  run=run / f"scout_{i}", extra=f"Search angle: {ang['angle']}",
                                                  label="bg-scout"), None
                except Exception as e:  # incl. budget refusals; one failed angle must not sink the chain
                    return i, ang, None, e
            # angles run in parallel: each high-effort scout is ~2 min of Gemini latency
            with ThreadPoolExecutor(max_workers=len(angles)) as ex:
                for i, ang, r, err in ex.map(_scout, enumerate(angles, 1)):
                    if isinstance(err, common.BudgetError):
                        stopped = stopped or f"during scouting: {err}"
                        continue
                    if err:
                        scouts.append({"angle": ang["angle"], "topic": ang["search_topic"], "n": 0, "run": "",
                                       "queries": [], "flags": [f"scout failed: {type(err).__name__}: {str(err)[:160]}"]})
                        continue
                    results.append(r["papers"])
                    scouts.append({"angle": ang["angle"], "topic": ang["search_topic"], "n": len(r["papers"]),
                                   "unverified": len(r["unverified"]), "run": r["run_dir"],
                                   "queries": r["search_queries_issued"], "flags": r["grounding_flags"]})
            papers = merge(results)
        for i, p in enumerate(papers, 1):
            p["sid"] = f"S{i}"
        if papers:
            tc.write_text(json.dumps({"topic": a.topic, "date": dt.date.today().isoformat(),
                                      "papers": [{k: v for k, v in p.items() if k != "sid"} for p in papers]},
                                     ensure_ascii=False))

        # 3. select
        if papers and not stopped:
            listing = "\n".join(
                f"{p['sid']}: {p['title']} ({p.get('year')}, {p.get('venue') or '?'}) angles={p['angles']} "
                f"open_full_text={'yes' if p.get('arxiv_id') else 'maybe'}\n    abstract: {(p.get('abstract') or 'n/a')[:900]}"
                for p in papers[:40])
            try:
                out = gemini_client.interact(_prompt("background_select", topic=a.topic, papers=listing, k=a.read + 2),
                                             effort=a.effort, label="bg-select", schema=SELECT_SCHEMA, tools=(),
                                             allow_over_daily=a.allow_over_daily, archive=run / "raw_select.json",
                                             reserve_usd=common.estimate_usd(len(listing), 3000))
            except common.BudgetError as e:
                out, stopped = None, f"before selecting papers: {e}"
            if out:
                by = {p["sid"]: p for p in papers}
                for s in (out["json"] or {}).get("selected") or []:
                    if s.get("id") in by and by[s["id"]] not in selected:
                        selected.append(dict(by[s["id"]], select_reason=s.get("reason")))
                if not selected:
                    selected = papers[: a.read + 2]

        # 4. read (parallel, stops at a.read successes)
        if selected and not stopped:
            cands = [p for p in selected if _ident(p)]

            def _one(p):
                try:
                    return p, modes.do_read(_ident(p), a.effort, None, a.allow_over_daily,
                                            run=run / f"read_{p['sid']}"), None
                except common.BudgetError as e:
                    return p, None, f"budget: {e}"
                except Exception as e:
                    return p, None, f"{type(e).__name__}: {str(e)[:160]}"
            queue = list(cands)
            while len(reads) < a.read and queue and not stopped:
                batch, queue = queue[: a.read - len(reads)], queue[a.read - len(reads):]
                with ThreadPoolExecutor(max_workers=3) as ex:
                    for p, r, err in ex.map(_one, batch):
                        if r:
                            reads.append((p, r))
                        else:
                            read_failed.append({"sid": p["sid"], "title": p["title"], "error": err})
                            if err.startswith("budget"):
                                stopped = stopped or f"during reading: {err}"

        # 5. synthesize
        ev_lines, ev_index = [], {}
        for j, (p, r) in enumerate(reads, 1):
            pid = f"P{j}"
            ev_lines.append(f"\n[{pid}] {p['title']} ({p.get('year')}) — full text read")
            for k, it in enumerate(r["items"], 1):
                eid = f"{pid}.E{k}"
                ev_index[eid] = {"paper": p["title"], "year": p.get("year"), "claim": it.get("claim"),
                                 "quote": it.get("quote"), "location": it.get("location"),
                                 "kind": it.get("kind"), "level": "full_text_quote"}
                ev_lines.append(f"{eid} [{it.get('dimension')}, {it.get('location')}] {it.get('claim')}\n"
                                f"   quote: \"{it.get('quote')}\"")
        read_titles = {p["title"] for p, _ in reads}
        for p in papers:
            if p["title"] in read_titles or not p.get("abstract"):
                continue
            ev_index[p["sid"]] = {"paper": p["title"], "year": p.get("year"), "level": "abstract_only"}
            ev_lines.append(f"\n[{p['sid']}] {p['title']} ({p.get('year')}) — abstract only\n   {p['abstract'][:900]}")
        previous = ""
        if prev and prev.get("brief_summary"):
            previous = f"\nPrevious brief on this topic ({prev['date']}), for continuity only:\n{prev['brief_summary']}\n"
        if ev_index and not (stopped and not reads):
            ev_text = "\n".join(ev_lines)
            try:
                out = gemini_client.interact(_prompt("background_synth", topic=a.topic, evidence=ev_text,
                                                     previous=previous),
                                             effort=a.effort, label="bg-synth", schema=SYNTH_SCHEMA, tools=(),
                                             allow_over_daily=a.allow_over_daily, archive=run / "raw_synth.json",
                                             reserve_usd=common.estimate_usd(len(ev_text), 6000), timeout=300)
                synth = out["json"] or {}
            except common.BudgetError as e:  # run cap or daily soft cap: keep what we have
                stopped = stopped or f"before synthesis: {e}"
    finally:
        spent = common.run_spent()
        common.set_run_cap(None)

    if synth:
        for sec in ("closest_work", "novelty_threats", "open_gaps"):
            for st in synth.get(sec) or []:
                ids = [e for e in st.get("evidence") or [] if e in ev_index]
                st["bad_evidence_ids"] = [e for e in st.get("evidence") or [] if e not in ev_index]
                st["evidence"] = ids
                st["support"] = ("full_text" if any(ev_index[e]["level"] == "full_text_quote" for e in ids)
                                 else "abstract_only" if ids else "none")
        if tc.exists():
            c = json.loads(tc.read_text())
            c["brief_summary"] = synth.get("summary")
            tc.write_text(json.dumps(c, ensure_ascii=False))
    r = {"mode": "background", "topic": a.topic, "effort": a.effort, "run_dir": str(run),
         "cost_usd": round(spent, 5), "max_cost": a.max_cost, "stopped_early": stopped, "log": log,
         "angles": angles, "scouts": scouts, "papers": papers,
         "read": [{"sid": p["sid"], "title": p["title"], "year": p.get("year"), "doi": p.get("doi"),
                   "arxiv_id": p.get("arxiv_id"), "select_reason": p.get("select_reason"),
                   "cached": bool(rr.get("cached")), "items_verified": len(rr["items"]),
                   "items_discarded": len(rr["discarded_items"]), "read_run": rr["run_dir"]} for p, rr in reads],
         "read_failed": read_failed, "evidence": ev_index, "brief": synth}
    r["report_md"] = render(r)
    r["budget_after"] = common.spend_summary()
    common.write_json(run / "result.json", r)
    (run / "background.md").write_text(r["report_md"])
    return r


def _ev(r, ids):
    out = []
    for e in ids:
        x = r["evidence"][e]
        if x["level"] == "full_text_quote":
            out.append(f"    - {e} {x['paper']} [{x.get('location')}]: \"{x['quote']}\"")
        else:
            out.append(f"    - {e} {x['paper']} (abstract only)")
    return out


def render(r: dict) -> str:
    L = [f"# background: {r['topic']}", "",
         f"effort={r['effort']}  cost=${r['cost_usd']:.4f} (cap ${r['max_cost']:.2f})  run: {r['run_dir']}"]
    if r["stopped_early"]:
        L.append(f"STOPPED EARLY {r['stopped_early']} — brief below is partial")
    L += r["log"]
    L.append(f"searches: {len(r['scouts'])}  papers found (verified, deduped): {len(r['papers'])}  "
             f"read in full: {len(r['read'])}")
    for s in r["scouts"]:
        L.append(f"- angle: {s['angle']} -> {s['n']} papers; queries: {'; '.join(s['queries'])[:200]}"
                 + (f"  WARNING {s['flags']}" if s["flags"] else ""))
    b = r["brief"]
    if b:
        if b.get("summary"):
            L += ["", "## Summary (Gemini's judgment)", b["summary"]]
        tag = {"full_text": "", "abstract_only": "  [abstract-level evidence only]",
               "none": "  [NO valid evidence ids — unsupported]"}
        L += ["", "## Closest existing work"]
        for c in b.get("closest_work") or []:
            L.append(f"- {c['paper']}: {c.get('similarity')}. Difference: {c.get('key_difference')}{tag[c['support']]}")
            L += _ev(r, c["evidence"])
        L += ["", "## Threats to novelty / known failure modes"]
        for c in b.get("novelty_threats") or []:
            L.append(f"- {c['statement']}{tag[c['support']]}")
            L += _ev(r, c["evidence"])
        L += ["", "## Gaps that look open (judgment; absence of evidence is not proof)"]
        for c in b.get("open_gaps") or []:
            L.append(f"- {c['gap']} — {c.get('why_it_looks_open')}{tag[c['support']]}")
            L += _ev(r, c["evidence"])
        if b.get("unresolved_questions"):
            L += ["", "## Unresolved"] + [f"- {q}" for q in b["unresolved_questions"]]
    else:
        L += ["", "No brief: synthesis did not run (see above)."]
    L += ["", "## Papers read in full (quotes verified)"]
    for p in r["read"]:
        L.append(f"- {p['sid']} {p['title']} ({p['year']}) doi={p['doi']} arXiv={p['arxiv_id']}  "
                 f"{p['items_verified']} quotes ok, {p['items_discarded']} discarded"
                 + ("  (cached)" if p["cached"] else ""))
    for f in r["read_failed"]:
        L.append(f"- NOT READ {f['sid']} {f['title']}: {f['error']}")
    others = [p for p in r["papers"] if p["sid"] not in {x["sid"] for x in r["read"]}]
    if others:
        L += ["", f"## Other verified papers found ({len(others)}, not read)"]
        for p in others[:30]:
            L.append(f"- {p['sid']} {p['title']} ({p.get('year')}) {p.get('venue') or ''} "
                     f"doi={p.get('doi')} arXiv={p.get('arxiv_id')} angles={p['angles']}")
    L += ["", "Evidence ids P#.E# are verbatim quotes from full text; S# are abstract-level. "
          "Unverified papers from each scout are in the scout_* run dirs."]
    return "\n".join(L)


def cmd_background(a) -> int:
    r = run_background(a)
    print(r["report_md"])
    return 0
