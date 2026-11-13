# Gemini Research Skill for Claude Code

A global [Claude Code](https://docs.claude.com/en/docs/claude-code) skill that gives Claude a fast, cheap literature-research tool backed by **Gemini 3.8 Flash + Google Search**, with an accuracy layer that does not trust the model: every paper is checked against arXiv / Crossref / OpenAlex, and every quote is checked against the paper's actual full text.

Claude can call it on its own (find related work, check what a paper really does, test a claim before relying on it), or you can run `/gemini-research ...` yourself.

## Why

- **Claude's own web research is slow.** Many sequential searches and page reads. Gemini Flash with server-side Google Search does a whole literature sweep in one call, for about a cent.
- **But LLM literature answers are unreliable.** Invented titles, wrong arXiv IDs, wrong years, quotes that don't exist, "supported" verdicts with nothing behind them. So the skill treats Gemini's output as *claims to check*:
  - Bibliographic fields come from arXiv / Crossref / OpenAlex, never from Gemini. Unknown titles are listed separately as *unverified*, not as results.
  - The script fetches the full text itself (arXiv HTML with LaTeX math, or PDF), gives Gemini exactly that text, and machine-checks every quote against it. Items whose quote isn't found are discarded.
  - Verdicts (`verify`, `audit`) only stand if backed by verified quotes; otherwise they are downgraded. Never upgraded.
  - The search queries Gemini actually issued are recorded; an answer with no search is flagged *ungrounded*.
- **Gemini gathers, Claude judges.** The output is labelled so Claude knows what is checked fact (bibliography, verbatim quotes) and what is Gemini's opinion (relevance, summaries, "gaps").

## Subcommands

| Command | What it does | Default effort | Measured* |
|---|---|---|---|
| `scout "<topic>"` | Gemini expands the topic into several Google queries and returns candidate papers; each is checked against the databases | low | 30 s, $0.008, ~8 queries |
| `read <arXiv id \| DOI \| URL> [--focus "q"]` | Reads one paper's public full text; extracts method / optimization / setup / evidence / limitations, each with a verified quote and location | medium | 35 s, $0.03 |
| `verify "<claim>" [--source ID]...` | Tests a claim against full-text sources (given, or found by search, deliberately including possible counter-evidence). Verdict: supported / contradicted / conflicting / insufficient | low | 10–40 s, $0.01–0.03 |
| `compare <ID> <ID> [...] [--dim "..."]` | 2–6 papers side by side; each cell quote-checked against its own paper | medium | 60 s, $0.07 (3 papers) |
| `audit <report.md\|.pdf\|.txt\|.html>` | Checks a deep-research report (e.g. exported from the Gemini app) against its own cited sources: bibliography, reachability, and each claim vs. the cited text | low | 25 s, $0.01 (7 claims) |
| `background "<topic>"` | Chained, all at `high`: search angles → parallel scouts → pick key papers → read them in full → brief (closest work, novelty threats, open gaps), every statement pointing at evidence ids | high | first run ≈ $0.70, ~120 queries, ~10 min; same topic again ≈ $0.12, 4 min (caches) |
| `budget` | Month's spend, caps, whether the key is set | — | free |

\* Measured 2026-11-11 with `gemini-3.8-flash` at $0.75 / $3.75 per 1M input / output tokens. Google's price list says token prices double on 2027-01-01.

All model commands take `--effort low|medium|high` (Gemini's thinking level).

Example `audit` result on a planted test report: a changed number ("72 cores" instead of 7,200) was marked *contradicted* with the original sentence quoted; a fabricated reference was marked *source unavailable / not in databases*; an unsupported "first ever" claim was marked *not found*; four true claims were *supported* with verbatim quotes.

## Install

Requirements: Linux or macOS, Python 3.10+, Claude Code, a Gemini API key ([Google AI Studio](https://aistudio.google.com/)). A paid-tier key is recommended: the free tier's Google Search quota is tiny, and free-tier content may be used to improve Google's products.

```bash
git clone https://github.com/hdeng26/claude-gemini-research-skill ~/.claude/skills/gemini-research
~/.claude/skills/gemini-research/install.sh
```

`install.sh` creates a private virtualenv in the skill directory, installs the dependencies, runs the offline tests, and creates `~/.gemini-research/config.env` (chmod 600). Put your key there:

```
GEMINI_API_KEY=...
```

Check it:

```bash
~/.claude/skills/gemini-research/.venv/bin/python ~/.claude/skills/gemini-research/scripts/research.py budget
```

Update: `git -C ~/.claude/skills/gemini-research pull && ~/.claude/skills/gemini-research/install.sh`

## Usage

In Claude Code, either let Claude use it when it needs literature, or ask directly:

```
/gemini-research scout "layerwise least-squares training without backpropagation"
/gemini-research read 1605.02026 --focus "Does it use gradients anywhere?"
/gemini-research verify "Direct feedback alignment fails to train convolutional networks"
/gemini-research compare 1605.02026 2109.03220 1609.01596
/gemini-research audit ~/Downloads/deep-research-report.pdf
/gemini-research background "closed-form layerwise training and how it degrades with depth"
```

Or run the script yourself:

```bash
R="$HOME/.claude/skills/gemini-research/.venv/bin/python $HOME/.claude/skills/gemini-research/scripts/research.py"
$R scout "feedback alignment depth scaling" --effort medium
$R verify "ADMM beats L-BFGS in final accuracy on Higgs" --source 1605.02026
```

Every run writes `~/.gemini-research/runs/<timestamp>-<mode>-<slug>/` with the raw Gemini responses, `result.json` (all evidence, abstracts, flags) and the printed report.

Exit codes: `0` ok, `2` budget refused, `3` Gemini call failed (incl. quota exhausted), `4` no public full text.

## How to read the output

- **Verified papers**: bibliographic fields are database values. `FLAGS` mean Gemini got a detail wrong (ID, year, authors).
- **Unverified / Could not check / Lookup failed**: don't cite. *Lookup failed* is a network problem, not evidence of fabrication.
- **Quotes** marked `(exact)` / `(fuzzy)` really are in the fetched text. That does **not** prove they support Gemini's sentence next to them; read the quote. Numbers in Gemini's sentence that aren't in the quote are unchecked.
- **Discarded** items had quotes not found in the source: likely fabricated, ignore.
- **Summaries, reasoning, caveats, differences, "gaps that look open"** are Gemini's judgment.
- **Absence of evidence is not novelty.** "N queries found nothing close" never means "this is new".

`SKILL.md` contains the full rules Claude follows.

## Budget and privacy

- Hard monthly cap (default **$15**), daily soft cap (**$2**, Claude must ask you before passing it), and a monthly Google-search query cap (**4,500**, inside Google's 5,000/month free search quota for Gemini 3+). Search queries usually run out before dollars do.
- `audit` and `background` also have a per-run cap (`--max-cost`, default $0.50 / $1.50); when a cap is hit the run stops and still reports what it finished.
- The model is `gemini-3.8-flash` by default. Set `GEMINI_MODEL=gemini-4-hydrogen` in `config.env` to switch; only models with a price table in `scripts/common.py` are accepted. In one measured low-effort scout (2026-11-13) Hydrogen took the same time and cost about 3-6x more.
- Costs are computed conservatively from the `usage` block of each response and written to `~/.gemini-research/ledger.jsonl`.
- Queries, claims and audited reports are sent to Google. `SKILL.md` tells Claude to describe topics in public, generic terms and never to send unpublished results, private code or draft text. Only public URLs can be read; there is deliberately no way to upload local files (except a report you pass to `audit`).
- Calls use `store=false`. OpenReview and some publishers block automated access; the skill reports those as *could not check* and never tries to get past bot checks.

## Known limitations

- Paywalled papers (many IEEE / Elsevier) can't be read in full, only their abstracts. When OpenAlex knows an open-access copy, that copy is used.
- Figures aren't seen: Gemini gets text (arXiv HTML or PDF text extraction).
- Quote checking proves a passage exists, not that it supports the claim; Claude is instructed to read the quotes, but that step relies on Claude.
- `audit` was tested on a hand-made report; the exact export formats of deep-research tools may need parser tweaks.
- `background` reads only ~5 papers in full; it is a starting point for planning, not a literature review.

## Layout

```
SKILL.md            what Claude reads: commands, when to call, accuracy rules
scripts/research.py CLI entry point
scripts/gemini_client.py  one Gemini interaction: budget check, retries, evidence kept (queries, URL status, usage)
scripts/scholar.py  arXiv / Crossref / OpenAlex / landing-page metadata checks
scripts/fulltext.py public full-text fetching (arXiv HTML/PDF, open-access PDFs, citation_pdf_url)
scripts/validate.py paper validation, quote matching, verdict rules
scripts/scout.py, modes.py (read/verify/compare), audit.py, background.py
prompts/            prompt templates
tests/              offline tests (no API calls); LIVE_SCHOLAR=1 adds live metadata checks
```

Run the tests: `.venv/bin/python -m pytest -q tests`

## License

MIT
