---
name: gemini-research
description: Fast literature research with Gemini 3.8 Flash. scout finds papers on a topic (Google Search), read extracts method/setup/evidence/limitations from one paper's full text, verify tests a claim against full-text sources, compare puts 2-6 papers side by side, audit checks a deep-research report against its own sources, background chains them into a brief to run before writing a research plan. Every paper is checked against arXiv/Crossref/OpenAlex and every quote against the fetched full text. Use it on your own when you need related work, need to know what a paper actually does, or need to check a research claim, instead of many slow web searches. Also runs when the user types /gemini-research.
argument-hint: "scout <topic> | read <arXiv|DOI|URL> | verify \"<claim>\" [--source ID] | compare <ID> <ID>... | audit <file> | background <topic> | budget"
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/research.py *)
---

# gemini-research

User arguments (if invoked by the user): `$ARGUMENTS`

Run everything through this one command:

```bash
${CLAUDE_SKILL_DIR}/.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/research.py <subcommand> ...
```

## Subcommands

| subcommand | what it does | default effort | measured (2026-11-11) |
|---|---|---|---|
| `scout "<topic>" [--max-papers 10]` | Gemini expands the topic into several Google queries and returns candidate papers; each is checked against arXiv/Crossref/OpenAlex | low | low 30 s $0.008 ~8 queries; medium 90 s $0.03 ~17 queries |
| `read <src> [--focus "question"]` | fetches the paper's public full text, Gemini extracts method / optimization / setup / evidence / limitations (+ your focus question), each with a quote checked against the text | medium | 35 s, $0.03 |
| `verify "<claim>" [--source <src>]... [--max-sources 3]` | tests a claim. With `--source`, reads those papers; without, searches for up to 3 sources (incl. ones that might refute) and reads them. Verdict: supported / contradicted / conflicting / insufficient | low | 10 s $0.01 with 1 source; 40 s $0.03 with search |
| `compare <src> <src> [...] [--dim "dimension"]...` | 2-6 papers side by side; each cell quote-checked against its own paper | medium | 60 s, $0.07 for 3 papers |
| `audit <report file> [--max-claims 40] [--max-cost 0.5]` | checks a deep-research report (e.g. exported from the Gemini app as .md/.pdf/.txt/.html) against its own cited sources: references checked, claims judged against the cited source text with verified quotes | low | 25 s, $0.01 for a 7-claim report |
| `background "<topic>" [--read 5] [--angles 4] [--max-cost 1.5] [--fresh]` | chained, all at `high`: splits the topic into search angles, scouts each, picks the key papers, reads them in full, writes a brief (closest work, novelty threats, open gaps) where every statement points at evidence ids | high | first run ≈ $0.70, ~120 Google queries, ~10 min (4 scouts in parallel ≈ 3.5 min); same topic again ≈ 4 min, $0.12 (caches) |
| `budget` | this month's spend, caps, whether the API key is set | — | free |

`<src>` = arXiv id (`1605.02026`), arXiv URL, DOI, or a public URL (HTML or PDF).
All take `--effort low|medium|high` (Gemini thinking level).

## When to call it on your own

- `background`: before you write a **research** plan (new experiment design, new direction,
  a paper's contribution/novelty). Not for engineering plans (refactors, scripts, plotting).
  Tell the user in one line that you are running it and its cap ($1.50 by default), then run it.
  Rerunning the same topic within 30 days only searches for new work and reuses read papers.
  Every plan statement that relies on background must point at an evidence id in the brief.
- `audit`: when the user hands you a deep-research report (Gemini app, etc.) or asks whether one is right.

- `scout`: you need related work or prior art, or are about to say "nobody has done X".
- `read`: you are about to describe what a specific paper does, uses, or shows.
- `verify`: you (or the user) are about to rely on a specific factual claim about the literature.
- `compare`: the user wants a side-by-side of known papers.
- Use `low` or `medium` yourself; `high` only when the user asks. Bash timeout 600000 ms.
- Don't rerun the same topic/paper: results are in `~/.gemini-research/runs/`, full texts are cached.

## What you may send (privacy)

Queries and claims go to Google. Describe things in public, generic terms. Never put
unpublished results, numbers, private code or draft text from the user's projects
into a topic, claim, focus question or dimension. Only public URLs can be read;
there is deliberately no way to upload local files.

## How to read the output (accuracy rules)

Bibliography:
- Verified papers' title/authors/year/venue/DOI/arXiv come from the databases, not Gemini. Safe to name.
- FLAGS mean Gemini got that detail wrong; use the database values.
- "Unverified" and "Could not check": never cite. You may say "Gemini also named X, which I could not verify".
- `scout`'s relevance/approach text is Gemini's unchecked opinion; read the abstract in `result.json` first.

Quotes (`read`, `verify`, `compare`):
- `(exact)` / `(fuzzy)` means the passage really is in the fetched text. It does **not** mean the
  passage supports Gemini's sentence above it. Before you repeat a claim, read the quote and
  check it says that. Numbers in Gemini's sentence that are not in the quote are unchecked.
- "Discarded" items had quotes not found in the text: treat them as possibly fabricated; ignore.
- `(inferred by Gemini)` items are Gemini's conclusions, not the authors' statements.
- `location_warning`: the cited Table/Eq./Section label does not appear in the text; the quote may still be right.
- Summaries, reasoning, caveats and "differences" are Gemini's judgment, unchecked.
- `reading_level=full_text_verified` means the full text was read and at least one quote matched.
  WARNING `very_short_text` means we probably only got an abstract or paywall page.
- Text comes from arXiv HTML (math kept as LaTeX) or PDF text extraction. Figures are not seen.

Verdicts (`verify`):
- The script only lets a verdict stand if it is backed by verified quotes; otherwise it is
  downgraded to `insufficient` and the report says so. It never upgrades.
- A verdict covers only the sources listed as read. `insufficient` is not evidence against the claim.
- Check that each quote is about the same thing as the claim (e.g. FA vs DFA, shallow vs deep).

Audit (`audit`):
- "Supported" / "Contradicted" need a quote found verbatim in the cited source. "Judgment without a matching quote" = treat as unchecked.
- "Not found in the cited source passages" means the locally selected passages did not settle it; the source may say it
  elsewhere. Say "not confirmed", not "false".
- "Cited source unavailable": paywalled, dead or fabricated; check the References section (`bib=not_in_databases`
  plus `not readable` is a strong sign the reference is fabricated).
- Only atomic research claims are extracted; generic statements may be skipped. Report text is sent
  to Google: only audit reports on public topics.

Background (`background`):
- Evidence ids P#.E# are verbatim full-text quotes; S# are abstract-only. A statement tagged
  `[abstract-level evidence only]` is weaker; `[NO valid evidence ids]` is unsupported: drop it.
- "Gaps that look open" are Gemini's judgment over a limited sample. Never turn one into
  "nobody has done X"; at most "the N papers found don't do X".
- `STOPPED EARLY` means the run cap or the daily soft cap was hit; the brief is partial. For the
  daily cap, ask the user before rerunning with `--allow-over-daily` (the rerun reuses caches).

General:
- `WARNING: ungrounded` means Gemini did not search; treat that list as unreliable.
- A short or empty result does not prove novelty. Say "N queries found nothing close", never "this is new".
- For anything going into a paper (formulas, numbers, exact claims), read the source yourself.

## Exit codes

- 0: ok, report printed; the run directory (path in the report) has `result.json` with everything.
- 2: **budget refused**. Tell the user the message verbatim. For the daily soft cap, ask the user;
  only if they agree, rerun with `--allow-over-daily`. Never raise caps yourself.
- 3: Gemini call failed (incl. quota exhausted). Report it; do not loop (the script retried once).
- 4: no public full text (paywalled / not found). Say so; don't substitute a summary from memory.

## Budget (in ~/.gemini-research/config.env)

`MONTHLY_CAP_USD=15` (hard), `DAILY_SOFT_CAP_USD=2`, `MONTHLY_SEARCH_QUERY_CAP=4500`
(stays inside Google's 5,000/month free search quota; it runs out before the dollar cap,
so prefer `low` for scout). Costs are computed conservatively from token usage. Token prices
double on 2027-01-01. Do not edit these values unless the user asks.

Model: `gemini-3.8-flash` by default; `GEMINI_MODEL=gemini-4-hydrogen` in config.env switches it
(measured 2026-11-13: same speed, ~3-6x the cost per low scout). Only models with a price table
in `scripts/common.py` are accepted. Do not switch models unless the user asks.
