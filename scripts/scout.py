"""scout: find candidate papers on a topic; every one checked against the databases."""
from __future__ import annotations

from pathlib import Path

import common
import gemini_client
import validate

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

SCOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "expanded_queries": {"type": "array", "items": {"type": "string"}},
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "year": {"type": ["integer", "null"]},
                    "arxiv_id": {"type": ["string", "null"]},
                    "doi": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                    "relevance": {"type": "string"},
                    "claimed_approach": {"type": "string"},
                    "found_via": {"type": "string"},
                },
                "required": ["title", "relevance"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["papers"],
}


def _prompt(name: str, **kw) -> str:
    text = (PROMPTS / f"{name}.md").read_text()
    for k, v in kw.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def do_scout(topic: str, effort: str, max_papers: int = 10, allow_over_daily: bool = False,
             run: Path | None = None, extra: str = "", label: str = "scout") -> dict:
    run = run or common.new_run_dir("scout", topic)
    run.mkdir(parents=True, exist_ok=True)
    prompt = _prompt("scout", topic=topic, max_papers=max_papers) + (("\n\n" + extra) if extra else "")
    calls = []
    parsed = gemini_client.interact(prompt, effort=effort, label=label, schema=SCOUT_SCHEMA,
                                    allow_over_daily=allow_over_daily, archive=run / "raw_1.json")
    calls.append(parsed)
    flags = validate.grounding_flags(parsed)
    if parsed["search_query_count"] == 0:
        # A literature answer without any search is from model memory: retry once, forcing search.
        parsed = gemini_client.interact(
            prompt + "\n\nYou did not search last time. You MUST call Google Search before answering.",
            effort=effort, label=label + "-retry", schema=SCOUT_SCHEMA,
            allow_over_daily=allow_over_daily, archive=run / "raw_2.json")
        calls.append(parsed)
        flags = validate.grounding_flags(parsed)
    data = parsed["json"] or {}
    claimed = (data.get("papers") or [])[: max_papers]
    checked = validate.validate_papers(claimed, parsed)
    result = {
        "mode": "scout", "topic": topic, "effort": effort, "model": common.MODEL,
        "run_dir": str(run),
        "cost_usd": round(sum(c["cost_usd"] for c in calls), 5),
        "search_queries_issued": parsed["search_queries"],
        "expanded_queries_claimed": data.get("expanded_queries") or [],
        "grounding_flags": flags,
        "papers": checked["papers"],
        "unverified": checked["unverified"],
        "web_citations": parsed["citations"],
        "url_results": parsed["url_results"],
        "model_notes": data.get("notes"),
        "budget_after": common.spend_summary(),
    }
    common.write_json(run / "result.json", result)
    (run / "report.md").write_text(render_scout(result))
    return result


def render_scout(r: dict) -> str:
    L = [f"# scout: {r['topic']}", ""]
    b = r["budget_after"]
    L.append(f"effort={r['effort']}  cost=${r['cost_usd']:.4f}  month=${b['month_usd']:.2f}  "
             f"search queries this month={b['month_search_queries']}")
    L.append(f"run: {r['run_dir']}")
    L.append("Google queries actually issued: " + ("; ".join(r["search_queries_issued"]) or "NONE"))
    for f in r["grounding_flags"]:
        L.append(f"WARNING: {f}")
    L += ["", f"## Verified papers ({len(r['papers'])})",
          "Bibliographic fields below come from arXiv/Crossref/OpenAlex, not from Gemini.", ""]
    for i, p in enumerate(r["papers"], 1):
        ids = ", ".join(x for x in (p.get("doi") and f"doi:{p['doi']}", p.get("arxiv_id") and f"arXiv:{p['arxiv_id']}") if x)
        authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        L.append(f"{i}. **{p['title']}** ({p.get('year')}) {authors.rstrip('.')}. {p.get('venue') or ''} {ids}")
        L.append(f"   reading_level={p['reading_level']}  metadata={p['metadata_source']}  url={p.get('url')}")
        if p["flags"]:
            L.append(f"   FLAGS (Gemini got these wrong): {'; '.join(p['flags'])}")
        mj = p["model_judgment"]
        if mj.get("relevance"):
            L.append(f"   model_judgment relevance: {mj['relevance']}")
        if mj.get("claimed_approach"):
            L.append(f"   model_judgment approach (unchecked; read the abstract): {mj['claimed_approach']}")
    manual = [u for u in r["unverified"] if any(f.startswith("landing_page_blocked") for f in u["flags"])]
    lookup = [u for u in r["unverified"] if any(f.startswith("metadata_lookup_error") for f in u["flags"])]
    rest = [u for u in r["unverified"] if u not in manual and u not in lookup]
    if lookup:
        L += ["", f"## Lookup failed ({len(lookup)}) — databases unreachable, NOT checked; not evidence of fabrication", ""]
        for u in lookup:
            c = u["claimed"]
            L.append(f"- {c.get('title')} ({c.get('year')}) {u['flags'][0]}")
    if manual:
        L += ["", f"## Could not check ({len(manual)}) — not in arXiv/OpenAlex/Crossref, and the page blocks automated access",
              "Treat as unverified: a fabricated title can sit on a real link. Cite only after someone opens "
              "the link and confirms this exact title.", ""]
        for u in manual:
            c = u["claimed"]
            L.append(f"- {c.get('title')} ({c.get('year')}) {c.get('url')}")
    L += ["", f"## Unverified ({len(rest)}) — do not cite", ""]
    for u in rest:
        c = u["claimed"]
        L.append(f"- {c.get('title')} ({c.get('year')}) claimed ids doi={c.get('doi')} arxiv={c.get('arxiv_id')} "
                 f"url={c.get('url')}; flags={u['flags']}; checked={u['checked']}")
    if r.get("model_notes"):
        L += ["", "## Model notes (model_judgment)", r["model_notes"]]
    L += ["", "Abstracts and full evidence: result.json in the run directory."]
    return "\n".join(L)


