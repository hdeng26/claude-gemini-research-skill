"""Fetch the public full text of a paper so Gemini reads exactly what we check quotes against.

Sources, in order:
  arXiv id/URL  -> arxiv.org/html/<id> (LaTeXML; math kept as LaTeX alttext) -> arxiv.org/pdf/<id>
  DOI           -> metadata via Crossref/OpenAlex; if it has an arXiv id use arXiv,
                   else OpenAlex's open-access PDF/landing page
  other URL     -> PDF if it is one, else the HTML page's text

Only public URLs are fetched; there is no way to pass local files. Texts are
cached under ~/.gemini-research/cache/fulltext/.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from html.parser import HTMLParser

import httpx

import common
import scholar

MAX_CHARS = 400_000  # ~100k tokens; longer texts are cut and flagged


class FullTextError(RuntimeError):
    pass


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg", "button"}
    BLOCK = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br",
             "table", "figcaption", "figure", "caption", "blockquote", "pre", "dd", "dt"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.in_math = [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self.skip += 1
            return
        if tag == "math":
            # LaTeXML keeps the source LaTeX in alttext; use it instead of MathML soup.
            if not self.skip:
                self.out.append(f" ${a.get('alttext', '')}$ ")
            self.in_math += 1
            return
        if tag in self.BLOCK:
            self.out.append("\n")
            if tag.startswith("h") and len(tag) == 2:
                self.out.append("\n## ")
        if tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(self.skip - 1, 0)
        elif tag == "math":
            self.in_math = max(self.in_math - 1, 0)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip and not self.in_math:
            self.out.append(data)


def html_to_text(page: str) -> str:
    m = re.search(r'<article[^>]*class="[^"]*ltx_document[^"]*"[^>]*>(.*)</article>', page, re.S)
    p = _HTMLText()
    p.feed(m.group(1) if m else page)
    text = "".join(p.out)
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


def pdf_to_text(data: bytes) -> str:
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, pg in enumerate(reader.pages, 1):
        try:
            pages.append(f"\n[page {i}]\n" + (pg.extract_text() or ""))
        except Exception:
            pages.append(f"\n[page {i}: text extraction failed]\n")
    return "".join(pages).strip()


def _get(url: str) -> httpx.Response:
    r = httpx.get(url, headers={"User-Agent": scholar.UA}, timeout=60, follow_redirects=True)
    return r


def _from_url(url: str) -> tuple[str, str]:
    r = _get(url)
    if r.status_code >= 400:
        raise FullTextError(f"{url}: http {r.status_code}")
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype or r.content[:5] == b"%PDF-":
        return pdf_to_text(r.content), "pdf"
    # Landing pages (PMLR, OpenReview, many journals) point at the PDF in a meta tag.
    m = re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)', r.text, re.I) or \
        re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url', r.text, re.I)
    if m:
        pdf_url = str(httpx.URL(str(r.url)).join(m.group(1)))
        pr = _get(pdf_url)
        if pr.status_code == 200 and pr.content[:5] == b"%PDF-":
            return pdf_to_text(pr.content), "landing_page_pdf"
    return html_to_text(r.text), "html"


def _arxiv(aid: str) -> tuple[str, str, str]:
    url = f"https://arxiv.org/html/{aid}"
    r = _get(url)
    if r.status_code == 200 and "ltx_document" in r.text:
        return html_to_text(r.text), "arxiv_html", url
    url = f"https://arxiv.org/pdf/{aid}"
    r = _get(url)
    if r.status_code == 200 and r.content[:5] == b"%PDF-":
        return pdf_to_text(r.content), "arxiv_pdf", url
    raise FullTextError(f"arXiv {aid}: no HTML or PDF (http {r.status_code})")


def classify(ident: str) -> dict:
    s = ident.strip()
    if re.match(r"^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$", s, re.I) or re.search(r"arxiv\.org/(abs|pdf|html)/", s, re.I):
        return {"arxiv_id": scholar.norm_arxiv(s)}
    if re.match(r"^(doi:|https?://(dx\.)?doi\.org/)?10\.\d{4,9}/\S+$", s, re.I):
        return {"doi": scholar.norm_doi(s)}
    if s.startswith("http"):
        return {"url": s}
    raise FullTextError(f"not an arXiv id, DOI or URL: {ident!r}")


def _cache_path(key: str):
    d = common.CACHE / "fulltext"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")


def get(ident: str) -> dict:
    """Return {"record", "text", "text_source", "text_url", "truncated", "chars", "flags"}."""
    ids = classify(ident)
    key = json.dumps(ids, sort_keys=True)
    cp = _cache_path(key)
    if cp.exists():
        return json.loads(cp.read_text())

    flags, record = [], None
    if ids.get("arxiv_id"):
        record = scholar.arxiv_by_id(ids["arxiv_id"])
        if not record:
            raise FullTextError(f"arXiv {ids['arxiv_id']} not found")
    elif ids.get("doi"):
        record = scholar.crossref_by_doi(ids["doi"]) or scholar.openalex_by_doi(ids["doi"])
        if not record:
            raise FullTextError(f"DOI {ids['doi']} not found in Crossref/OpenAlex")
        oa = scholar.openalex_by_doi(ids["doi"]) or {}
        record.setdefault("arxiv_id", None)
        record["arxiv_id"] = record.get("arxiv_id") or oa.get("arxiv_id")
        record["oa_pdf"] = oa.get("oa_pdf")
        if not record.get("abstract") and oa.get("abstract"):
            record["abstract"] = oa["abstract"]

    if record and record.get("arxiv_id"):
        text, src, url = _arxiv(record["arxiv_id"])
    elif record and record.get("oa_pdf"):
        url = record["oa_pdf"]
        text, src = _from_url(url)
        src = "open_access_" + src
    elif ids.get("url"):
        url = ids["url"]
        text, src = _from_url(url)
        lp = scholar.landing_page(url)
        record = lp.get("record") or {"source": "url_only", "title": None, "authors": [], "year": None,
                                      "doi": None, "arxiv_id": None, "url": url}
        if record["source"] == "url_only":
            flags.append("no_citation_metadata_on_page: title/authors unverified")
    else:
        raise FullTextError(f"{ident}: no open-access full text found (paywalled?)")

    if len(text) < 3000:
        flags.append(f"very_short_text ({len(text)} chars): probably an abstract page or paywall, not the paper")
    truncated = len(text) > MAX_CHARS
    if truncated:
        text = text[:MAX_CHARS]
        flags.append(f"text_truncated_at_{MAX_CHARS}_chars")
    out = {"record": record, "text": text, "text_source": src, "text_url": url,
           "truncated": truncated, "chars": len(text), "flags": flags}
    if "very_short_text" not in " ".join(flags):
        cp.write_text(json.dumps(out, ensure_ascii=False))
    return out
