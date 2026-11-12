"""Bibliographic ground truth from arXiv, Crossref and OpenAlex (no LLM involved).

`resolve(paper)` takes whatever Gemini claimed (title, authors, year, doi,
arxiv_id) and returns the canonical record if one of the databases has a work
whose title matches, plus flags for anything Gemini got wrong.
"""
from __future__ import annotations

import difflib
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx

UA = "gemini-research/0.1 (scholarly metadata check)"
TITLE_MATCH = 0.90
_last_call: dict[str, float] = {}
import threading as _threading
_slot_lock = _threading.Lock()


def _reserve_slot(host: str):
    """Thread-safe per-host spacing: each caller gets its own time slot, then sleeps until it."""
    with _slot_lock:
        t = max(time.time(), _last_call.get(host, 0) + _MIN_GAP.get(host, 0))
        _last_call[host] = t
    if t > time.time():
        time.sleep(t - time.time())
_MIN_GAP = {"export.arxiv.org": 3.0, "api.crossref.org": 0.2, "api.openalex.org": 0.15}


def _get(url: str, params: dict | None = None) -> httpx.Response | None:
    """GET with per-host spacing; backs off on 429/5xx (honours Retry-After, max 60 s)."""
    host = httpx.URL(url).host
    # arXiv has an OpenAlex fallback, so give up on it quickly; others get a longer backoff.
    delays = [5] if host == "export.arxiv.org" else [5, 15, 30]
    for attempt in range(len(delays) + 1):
        _reserve_slot(host)
        try:
            r = httpx.get(url, params=params, headers={"User-Agent": UA}, timeout=20, follow_redirects=True)
        except httpx.HTTPError:
            if attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            raise
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503, 504) and attempt < len(delays):
            ra = r.headers.get("retry-after", "")
            time.sleep(min(int(ra), 60) if ra.isdigit() else delays[attempt])
            continue
        r.raise_for_status()
        return r
    return None


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode()
    t = re.sub(r"\$[^$]*\$", " ", t)  # drop inline LaTeX
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", t.lower())).strip()


def title_sim(a: str, b: str) -> float:
    a, b = norm_title(a), norm_title(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def surname(name: str) -> str:
    name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().strip()
    if "," in name:
        return name.split(",")[0].strip().lower()
    parts = name.split()
    return parts[-1].lower() if parts else ""


def norm_doi(d: str | None) -> str | None:
    if not d:
        return None
    d = d.strip().lower()
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d)
    return d or None


def norm_arxiv(a: str | None) -> str | None:
    if not a:
        return None
    a = a.strip()
    a = re.sub(r"^(https?://)?(www\.)?arxiv\.org/(abs|pdf|html)/", "", a, flags=re.I)
    a = re.sub(r"^arxiv:\s*", "", a, flags=re.I)
    a = re.sub(r"\.pdf$", "", a)
    a = re.sub(r"v\d+$", "", a)
    return a or None


# --- arXiv -----------------------------------------------------------------
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _arxiv_entries(params: dict) -> list[dict]:
    r = _get("https://export.arxiv.org/api/query", params)
    if r is None:
        return []
    root = ET.fromstring(r.text)
    out = []
    for e in root.findall("a:entry", _NS):
        eid = (e.findtext("a:id", "", _NS) or "").strip()
        if "/abs/" not in eid:
            continue  # arXiv returns an error entry for bad ids
        doi = e.findtext("arxiv:doi", None, _NS)
        out.append({
            "source": "arxiv",
            "title": re.sub(r"\s+", " ", e.findtext("a:title", "", _NS)).strip(),
            "authors": [a.findtext("a:name", "", _NS) for a in e.findall("a:author", _NS)],
            "year": int((e.findtext("a:published", "", _NS) or "0000")[:4]) or None,
            "arxiv_id": norm_arxiv(eid.split("/abs/")[-1]),
            "doi": norm_doi(doi),
            "url": f"https://arxiv.org/abs/{norm_arxiv(eid.split('/abs/')[-1])}",
            "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", "", _NS)).strip() or None,
            "venue": e.findtext("arxiv:journal_ref", None, _NS) or "arXiv preprint",
        })
    return out


def arxiv_by_id(aid: str) -> dict | None:
    """OpenAlex first (indexes arXiv as DOI 10.48550/arXiv.<id>, no 3 s spacing), arXiv API as fallback
    for very new papers or when OpenAlex lacks the abstract."""
    rec = None
    try:
        rec = openalex_by_doi(f"10.48550/arxiv.{aid}")
    except httpx.HTTPError:
        rec = None
    if rec and rec.get("title") and rec.get("abstract"):
        rec["arxiv_id"] = aid
        rec["url"] = f"https://arxiv.org/abs/{aid}"
        return rec
    try:
        hits = _arxiv_entries({"id_list": aid, "max_results": 1})
        if hits:
            return hits[0]
    except httpx.HTTPError:
        pass
    if rec and rec.get("title"):
        rec["arxiv_id"] = aid
        rec["url"] = f"https://arxiv.org/abs/{aid}"
    return rec if rec and rec.get("title") else None


def arxiv_search_title(title: str, n: int = 5) -> list[dict]:
    words = norm_title(title).split()[:12]
    if not words:
        return []
    q = " AND ".join(f"ti:{w}" for w in words if len(w) > 2) or f'ti:"{title}"'
    try:
        return _arxiv_entries({"search_query": q, "max_results": n})
    except httpx.HTTPError:
        return []  # supplementary to the OpenAlex title search, which ran first


# --- OpenAlex --------------------------------------------------------------
def _openalex_record(w: dict) -> dict:
    inv = w.get("abstract_inverted_index") or {}
    abstract = None
    if inv:
        pos = {}
        for word, idxs in inv.items():
            for i in idxs:
                pos[i] = word
        abstract = " ".join(pos[i] for i in sorted(pos))
    ids = w.get("ids") or {}
    arxiv = None
    for loc in w.get("locations") or []:
        u = (loc.get("landing_page_url") or "")
        if "arxiv.org/abs/" in u:
            arxiv = norm_arxiv(u)
            break
    src = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "source": "openalex",
        "openalex_id": w.get("id"),
        "title": w.get("display_name") or w.get("title") or "",
        "authors": [(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []],
        "year": w.get("publication_year"),
        "doi": norm_doi(w.get("doi") or ids.get("doi")),
        "arxiv_id": arxiv,
        "url": (w.get("primary_location") or {}).get("landing_page_url") or w.get("doi"),
        "abstract": abstract,
        "venue": src.get("display_name"),
        "cited_by_count": w.get("cited_by_count"),
        "oa_pdf": (w.get("best_oa_location") or {}).get("pdf_url"),
    }


def openalex_by_doi(doi: str) -> dict | None:
    r = _get(f"https://api.openalex.org/works/doi:{quote(doi, safe='/')}")
    return _openalex_record(r.json()) if r is not None else None


def openalex_search_title(title: str, n: int = 5) -> list[dict]:
    q = norm_title(title)
    if not q:
        return []
    r = _get("https://api.openalex.org/works", {"search": q, "per-page": n})
    return [_openalex_record(w) for w in (r.json().get("results") or [])] if r is not None else []


# --- Crossref --------------------------------------------------------------
def _crossref_record(m: dict) -> dict:
    title = (m.get("title") or [""])[0]
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        parts = (m.get(k) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    return {
        "source": "crossref",
        "title": re.sub(r"<[^>]+>", "", title),
        "authors": [" ".join(x for x in (a.get("given"), a.get("family")) if x) for a in m.get("author") or []],
        "year": year,
        "doi": norm_doi(m.get("DOI")),
        "arxiv_id": None,
        "url": m.get("URL"),
        "abstract": re.sub(r"<[^>]+>", "", m["abstract"]).strip() if m.get("abstract") else None,
        "venue": (m.get("container-title") or [None])[0],
        "type": m.get("type"),
    }


def crossref_by_doi(doi: str) -> dict | None:
    r = _get(f"https://api.crossref.org/works/{quote(doi, safe='/')}")
    return _crossref_record(r.json()["message"]) if r is not None else None


# --- landing page (weakest source) ------------------------------------------
_CHALLENGE = re.compile(r"challenge|captcha|cf-chl|are you a robot|verify you are human", re.I)


def landing_page(url: str) -> dict:
    """Read Highwire `citation_*` meta tags from the page Gemini cited.

    Never tries to get past a bot challenge: a blocked page is reported as
    blocked so a human (or Claude, by opening it) can check it.
    """
    try:
        r = httpx.get(url, headers={"User-Agent": UA}, timeout=20, follow_redirects=True)
    except httpx.HTTPError as e:
        return {"blocked": False, "error": f"{type(e).__name__}"}
    if r.status_code in (401, 403, 429) or (r.status_code == 200 and _CHALLENGE.search(r.text[:3000])
                                             and "citation_title" not in r.text):
        return {"blocked": True, "status_code": r.status_code}
    if r.status_code >= 400:
        return {"blocked": False, "error": f"http {r.status_code}"}
    metas = re.findall(r"<meta\s+[^>]*>", r.text[:300000], re.I)
    tags: dict[str, list[str]] = {}
    for m in metas:
        n = re.search(r'name\s*=\s*["\']([^"\']+)["\']', m, re.I)
        c = re.search(r'content\s*=\s*["\']([^"\']*)["\']', m, re.I)
        if n and c:
            tags.setdefault(n.group(1).lower(), []).append(c.group(1))
    t = (tags.get("citation_title") or [None])[0]
    if not t:
        return {"blocked": False, "error": "no citation_title meta"}
    date = (tags.get("citation_publication_date") or tags.get("citation_date") or [""])[0]
    yr = re.search(r"(19|20)\d\d", date)
    import html as _html
    return {"blocked": False, "record": {
        "source": "landing_page",
        "title": _html.unescape(t),
        "authors": [_html.unescape(a) for a in tags.get("citation_author") or []],
        "year": int(yr.group(0)) if yr else None,
        "doi": norm_doi((tags.get("citation_doi") or [None])[0]),
        "arxiv_id": norm_arxiv((tags.get("citation_arxiv_id") or [None])[0]),
        "url": str(r.url),
        "abstract": _html.unescape((tags.get("citation_abstract") or [""])[0]) or None,
        "venue": (tags.get("citation_conference_title") or tags.get("citation_journal_title") or [None])[0],
    }}


# --- resolution ------------------------------------------------------------
def resolve(claim: dict) -> dict:
    """Check one Gemini-claimed paper against the databases.

    Returns {"status": "verified"|"unverified", "record": canonical|None,
             "flags": [...], "title_similarity": float, "checked": [...]}.
    """
    title = claim.get("title") or ""
    doi = norm_doi(claim.get("doi"))
    aid = norm_arxiv(claim.get("arxiv_id"))
    flags, checked = [], []

    def good(rec):
        return rec and title_sim(title, rec["title"]) >= TITLE_MATCH

    found = None
    # 1. Identifiers the model gave. An id whose title does not match is a red flag.
    if aid:
        checked.append(f"arxiv:{aid}")
        rec = arxiv_by_id(aid)
        if good(rec):
            found = rec
        elif rec:
            flags.append(f"arxiv_id_points_to_other_paper: {rec['title'][:80]}")
        else:
            flags.append("arxiv_id_not_found")
    if doi and not found:
        checked.append(f"doi:{doi}")
        rec = crossref_by_doi(doi) or openalex_by_doi(doi)
        if good(rec):
            found = rec
        elif rec:
            flags.append(f"doi_points_to_other_paper: {rec['title'][:80]}")
        else:
            flags.append("doi_not_found")
    # 2. Title search.
    if not found and title:
        checked.append("openalex:title")
        cands = openalex_search_title(title)
        if not any(good(c) for c in cands):
            checked.append("arxiv:title")
            cands += arxiv_search_title(title)
        best = max(cands, key=lambda c: title_sim(title, c["title"]), default=None)
        if good(best):
            found = best

    # 3. Not in any database: the cited page itself, if it carries citation meta.
    if not found and (claim.get("url") or "").startswith("http"):
        checked.append("landing_page")
        lp = landing_page(claim["url"])
        if lp.get("blocked"):
            flags.append(f"landing_page_blocked_by_bot_check: verify manually at {claim['url']}")
        elif good(lp.get("record")):
            found = lp["record"]
            flags.append("not_in_scholarly_databases: metadata from the paper's own page only")
        elif lp.get("record"):
            flags.append(f"landing_page_shows_other_title: {lp['record']['title'][:80]}")

    if not found:
        if not flags:
            flags = ["no_database_match"]
        elif not any(f.startswith(("landing_page_blocked",)) for f in flags):
            flags.append("no_database_match")
        return {"status": "unverified", "record": None, "flags": flags,
                "title_similarity": 0.0, "checked": checked}

    # 3. Enrich: get an abstract and a DOI if the first source lacked them.
    if found.get("doi") and not found.get("abstract"):
        extra = openalex_by_doi(found["doi"])
        if extra and extra.get("abstract"):
            found["abstract"] = extra["abstract"]
            found.setdefault("openalex_id", extra.get("openalex_id"))
    # 4. Compare the rest of what the model claimed with the canonical record.
    if doi and found.get("doi") and doi != found["doi"]:
        flags.append(f"claimed_doi_wrong: claimed {doi}, actual {found['doi']}")
    if aid and found.get("arxiv_id") and aid != found["arxiv_id"]:
        flags.append(f"claimed_arxiv_id_wrong: claimed {aid}, actual {found['arxiv_id']}")
    cy = claim.get("year")
    if cy and found.get("year") and abs(int(cy) - int(found["year"])) > 1:
        flags.append(f"claimed_year_wrong: claimed {cy}, actual {found['year']}")
    claimed_auth = {surname(a) for a in claim.get("authors") or [] if surname(a)}
    real_auth = {surname(a) for a in found.get("authors") or [] if surname(a)}
    if claimed_auth and real_auth and not (claimed_auth & real_auth):
        flags.append("claimed_authors_wrong: no surname overlap")
    return {"status": "verified", "record": found, "flags": flags,
            "title_similarity": round(title_sim(title, found["title"]), 3), "checked": checked}
