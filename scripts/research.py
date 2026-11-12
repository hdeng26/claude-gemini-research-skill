#!/usr/bin/env python3
"""gemini-research CLI.

  research.py scout "<topic>" [--effort low|medium|high] [--max-papers 10]
  research.py read <arXiv id | DOI | URL> [--focus "question"]
  research.py verify "<claim>" [--source ID ...]
  research.py compare <ID> <ID> [...] [--dim "dimension" ...]
  research.py audit <report.md|.pdf|.txt|.html> [--max-claims 40] [--max-cost 0.5]
  research.py background "<topic>" [--read 5] [--max-cost 1.5] [--fresh]
  research.py budget

Every run writes ~/.gemini-research/runs/<id>/{raw_*.json,result.json,report.md}
and prints the report. Exit codes: 0 ok, 2 budget refused, 3 API/other failure, 4 no full text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gemini_client  # noqa: E402
import validate  # noqa: E402
import modes  # noqa: E402
import scout  # noqa: E402
import audit  # noqa: E402
import background  # noqa: E402

def cmd_scout(a) -> int:
    r = scout.do_scout(a.topic, a.effort, a.max_papers, a.allow_over_daily)
    print(scout.render_scout(r))
    return 0


def cmd_budget(a) -> int:
    cfg = common.load_config()
    s = common.spend_summary()
    print(json.dumps({**s, "monthly_cap_usd": float(cfg["MONTHLY_CAP_USD"]),
                      "daily_soft_cap_usd": float(cfg["DAILY_SOFT_CAP_USD"]),
                      "monthly_search_query_cap": int(cfg["MONTHLY_SEARCH_QUERY_CAP"]),
                      "prices_per_1M": common.prices(),
                      "api_key_configured": bool(cfg.get("GEMINI_API_KEY"))}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="research.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scout", help="find candidate papers on a topic, metadata-verified")
    s.add_argument("topic")
    s.add_argument("--effort", choices=["low", "medium", "high"], default="low")
    s.add_argument("--max-papers", type=int, default=10)
    s.add_argument("--allow-over-daily", action="store_true",
                   help="only after the user agreed to pass the daily soft cap")
    s.set_defaults(fn=cmd_scout)
    def common_opts(p, effort):
        p.add_argument("--effort", choices=["low", "medium", "high"], default=effort)
        p.add_argument("--allow-over-daily", action="store_true",
                       help="only after the user agreed to pass the daily soft cap")
    r = sub.add_parser("read", help="read one paper's full text; every extracted item quote-checked")
    r.add_argument("source", help="arXiv id/URL, DOI, or public URL (HTML or PDF)")
    r.add_argument("--focus", help="a specific question to answer about the paper")
    common_opts(r, "medium")
    r.set_defaults(fn=modes.cmd_read)
    v = sub.add_parser("verify", help="test a claim against full-text sources; verdict needs verified quotes")
    v.add_argument("claim")
    v.add_argument("--source", action="append", help="arXiv id/DOI/URL; repeatable. Omit to search for sources")
    v.add_argument("--max-sources", type=int, default=3)
    common_opts(v, "low")
    v.set_defaults(fn=modes.cmd_verify)
    c = sub.add_parser("compare", help="compare 2-6 papers along dimensions; every cell quote-checked")
    c.add_argument("sources", nargs="+")
    c.add_argument("--dim", dest="dims", action="append", help="comparison dimension; repeatable")
    common_opts(c, "medium")
    c.set_defaults(fn=modes.cmd_compare)
    au = sub.add_parser("audit", help="check a deep-research report (md/txt/pdf/html) against its own sources")
    au.add_argument("report", help="path to the exported report")
    au.add_argument("--max-claims", type=int, default=40)
    au.add_argument("--max-cost", type=float, default=0.50, help="USD cap for this run")
    au.add_argument("--batch", type=int, default=8, help="claims per checking call")
    common_opts(au, "low")
    au.set_defaults(fn=audit.cmd_audit)
    bg = sub.add_parser("background", help="scout several angles, read the key papers, write a background brief")
    bg.add_argument("topic")
    bg.add_argument("--angles", type=int, default=4)
    bg.add_argument("--read", type=int, default=5, help="papers to read in full")
    bg.add_argument("--max-cost", type=float, default=1.50, help="USD cap for this run")
    bg.add_argument("--reuse-days", type=int, default=30, help="reuse this topic's earlier search if newer than this")
    bg.add_argument("--fresh", action="store_true", help="ignore the topic cache")
    common_opts(bg, "high")
    bg.set_defaults(fn=background.cmd_background)
    sub.add_parser("budget", help="show this month's spend and caps").set_defaults(fn=cmd_budget)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except common.BudgetError as e:
        print(f"BUDGET REFUSED: {e}", file=sys.stderr)
        return 2
    except gemini_client.GeminiError as e:
        print(f"GEMINI FAILED: {e}", file=sys.stderr)
        return 3
    except modes.fulltext.FullTextError as e:
        print(f"FULL TEXT UNAVAILABLE: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
