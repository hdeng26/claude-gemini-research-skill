"""Turn Gemini's claimed papers into evidence entries. No LLM calls here.

Rules (the skill's accuracy contract):
  - bibliographic fields come from arXiv/Crossref/OpenAlex, never from Gemini
  - a paper no database knows is `unverified` and kept out of the main list
  - reading_level is decided here, not by the model:
      search_result_only  Gemini named it; we only have its metadata
      abstract_verified   metadata matched AND we hold the real abstract
      full_text_verified  (read mode only) quotes matched against the full text
  - anything Gemini says about a paper's content is labelled model_judgment
"""
from __future__ import annotations

import scholar


def grounding_flags(parsed: dict) -> list[str]:
    flags = []
    if parsed["search_query_count"] == 0:
        flags.append("ungrounded: the model issued no Google Search queries")
    if parsed["errors"]:
        flags += parsed["errors"]
    return flags


def _key(rec: dict) -> str:
    return (rec.get("doi") or (rec.get("arxiv_id") and f"arxiv:{rec['arxiv_id']}")
            or rec.get("openalex_id") or scholar.norm_title(rec.get("title", "")))


def validate_papers(claimed: list[dict], parsed: dict | None = None) -> dict:
    url_status = {r["url"]: r["status"] for r in (parsed or {}).get("url_results", []) if r.get("url")}
    verified, unverified, seen = [], [], {}
    for c in claimed:
        model_view = {k: c.get(k) for k in ("title", "authors", "year", "doi", "arxiv_id", "url")}
        try:
            res = scholar.resolve(c)
        except Exception as e:  # network trouble is not evidence of fabrication
            unverified.append({
                "claimed": model_view, "checked": [], "model_judgment": {},
                "flags": [f"metadata_lookup_error: {type(e).__name__}: {str(e)[:120]}"],
                "note": "Lookup failed (network/rate limit). Not checked, not necessarily fabricated; rerun later.",
            })
            continue
        judgment = {k: c[k] for k in ("relevance", "claimed_approach", "found_via") if c.get(k)}
        if res["status"] != "verified":
            unverified.append({
                "claimed": model_view, "flags": res["flags"], "checked": res["checked"],
                "model_judgment": judgment,
                "note": "No database match. Do not cite; may be fabricated or mis-titled.",
            })
            continue
        rec = res["record"]
        k = _key(rec)
        if k in seen:
            seen[k]["duplicates_merged"] = seen[k].get("duplicates_merged", 0) + 1
            continue
        entry = {
            "title": rec["title"], "authors": rec.get("authors") or [], "year": rec.get("year"),
            "venue": rec.get("venue"), "doi": rec.get("doi"), "arxiv_id": rec.get("arxiv_id"),
            "url": rec.get("url"), "metadata_source": rec["source"],
            "source_status": "verified",
            "reading_level": "abstract_verified" if rec.get("abstract") else "search_result_only",
            "abstract": rec.get("abstract"),
            "flags": res["flags"],
            "title_similarity": res["title_similarity"],
            "model_judgment": judgment,
            "model_claimed": model_view,
        }
        if c.get("url") and c["url"] in url_status:
            entry["url_context_status"] = url_status[c["url"]]
        seen[k] = entry
        verified.append(entry)
    return {"papers": verified, "unverified": unverified}


# ---------------------------------------------------------------------------
# Quote checking (read / verify / compare)
# ---------------------------------------------------------------------------
import re as _re
import unicodedata as _ud

MIN_QUOTE_WORDS = 6
FUZZY_CUTOFF = 90


def norm_text(s: str) -> str:
    s = _ud.normalize("NFKD", s or "")
    s = _re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)  # undo end-of-line hyphenation
    s = "".join(ch for ch in s if not _ud.combining(ch)).lower()
    s = _re.sub(r"[^a-z0-9]+", " ", s)
    return _re.sub(r"\s+", " ", s).strip()


def match_quote(quote: str, text_norm: str) -> dict:
    """exact / fuzzy (>= FUZZY_CUTOFF) / not_found / too_short."""
    q = norm_text(quote)
    if len(q.split()) < MIN_QUOTE_WORDS:
        return {"status": "too_short", "score": None}
    if q in text_norm:
        return {"status": "exact", "score": 100.0}
    from rapidfuzz import fuzz
    res = fuzz.partial_ratio_alignment(q, text_norm, score_cutoff=FUZZY_CUTOFF)
    if res is not None:
        return {"status": "fuzzy", "score": round(res.score, 1)}
    return {"status": "not_found", "score": None}


_LOC = _re.compile(r"\b(table|tab|figure|fig|equation|eq|section|sec|appendix|algorithm|alg|theorem|lemma|proposition|corollary)\.?\s*\(?([a-z]?\d+(?:\.\d+)*|[a-z])\)?", _re.I)
_CANON = {"tab": "table", "fig": "figure", "eq": "equation", "sec": "section", "alg": "algorithm"}


def check_location(location: str, text: str) -> str | None:
    """Warn when a cited label (e.g. 'Table 3') does not occur in the text at all."""
    missing = []
    low = text.lower()
    for kind, num in _LOC.findall(location or ""):
        kind = _CANON.get(kind.lower(), kind.lower())
        if kind == "equation":
            ok = _re.search(rf"\({_re.escape(num)}\)|eq(uation)?\.?\s*\(?{_re.escape(num)}\b|\\tag\{{{_re.escape(num)}\}}", low)
        elif kind == "section":
            ok = _re.search(rf"(^|\n|\s){_re.escape(num)}\.?\s+[a-z]", low) or _re.search(rf"section\s*{_re.escape(num)}\b", low)
        else:
            ok = _re.search(rf"{kind}\s*{_re.escape(num.lower())}\b", low)
        if not ok:
            missing.append(f"{kind} {num}")
    return f"location_label_not_found_in_text: {', '.join(missing)}" if missing else None


def check_items(items: list[dict], texts: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Split model items into (verified, unsupported) by checking each quote.

    texts: paper_id -> {"text": raw, "norm": normalized}. An item without a
    paper_id is checked against the only paper if there is exactly one.
    """
    ok, bad = [], []
    only = next(iter(texts)) if len(texts) == 1 else None
    for it in items:
        pid = it.get("paper_id") or only
        t = texts.get(pid)
        it = dict(it, paper_id=pid)
        if not t:
            it["quote_status"] = "unknown_paper_id"
            bad.append(it)
            continue
        m = match_quote(it.get("quote") or "", t["norm"])
        it["quote_status"], it["quote_score"] = m["status"], m["score"]
        warn = check_location(it.get("location") or "", t["text"])
        if warn:
            it["location_warning"] = warn
        (ok if m["status"] in ("exact", "fuzzy") else bad).append(it)
    return ok, bad


def final_verdict(model_verdict: str, verified_evidence: list[dict]) -> tuple[str, str | None]:
    """Only verified quotes can carry a verdict. Never upgrades, only downgrades."""
    sup = any(e.get("stance") == "supports" for e in verified_evidence)
    con = any(e.get("stance") == "contradicts" for e in verified_evidence)
    v = model_verdict
    if v == "supported":
        if not sup:
            return "insufficient", "downgraded: no verified supporting quote"
        if con:
            return "conflicting", "changed: verified quotes on both sides"
    elif v == "contradicted":
        if not con:
            return "insufficient", "downgraded: no verified contradicting quote"
        if sup:
            return "conflicting", "changed: verified quotes on both sides"
    elif v == "conflicting":
        if not (sup and con):
            return "insufficient", "downgraded: conflict needs verified quotes on both sides"
    elif v != "insufficient":
        return "insufficient", f"unknown verdict {v!r}"
    return v, None
