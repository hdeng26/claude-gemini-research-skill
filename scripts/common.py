"""Shared paths, config, pricing and the monthly budget ledger.

Everything persistent lives under GEMINI_RESEARCH_HOME (default ~/.gemini-research):
  config.env    GEMINI_API_KEY=... (chmod 600, written by the user)
  ledger.jsonl  one line per billed API call
  runs/         one directory per research run (raw responses, results, report)
  cache/        reserved for paper/topic caches
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

HOME = Path(os.environ.get("GEMINI_RESEARCH_HOME", Path.home() / ".gemini-research"))
CONFIG = HOME / "config.env"
LEDGER = HOME / "ledger.jsonl"
RUNS = HOME / "runs"
CACHE = HOME / "cache"

MODEL = "gemini-3.8-flash"

# Budget limits (USD / queries). Override in config.env.
DEFAULTS = {
    "MONTHLY_CAP_USD": "15",       # hard: calls are refused past this
    "DAILY_SOFT_CAP_USD": "2",     # soft: refused unless --allow-over-daily
    "MONTHLY_SEARCH_QUERY_CAP": "4500",  # stay inside the 5,000/month free search quota
}

# Worst-case cost reserved before a call, by effort. A call is refused if
# month_spend + reserve would pass the monthly cap. Re-tune after real runs.
RESERVE_USD = {"low": 0.05, "medium": 0.10, "high": 0.20}  # measured max so far: $0.07 (compare, medium)

# Prices per 1M tokens, Standard tier, from the Gemini API pricing page
# (checked 2026-11-11). Token prices double on 2027-01-01.
PRICE_TABLE = [
    # (valid_from, input, output_incl_thinking, cached_input)
    (dt.date(2000, 1, 1), 0.75, 3.75, 0.075),
    (dt.date(2027, 1, 1), 1.50, 7.50, 0.15),
]
SEARCH_PRICE_PER_QUERY = 14 / 1000  # only past the free quota; see ledger notes


def prices(today: dt.date | None = None):
    today = today or dt.date.today()
    row = [r for r in PRICE_TABLE if r[0] <= today][-1]
    return {"input": row[1], "output": row[2], "cached": row[3]}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG.exists():
        for line in CONFIG.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    for k in list(DEFAULTS) + ["GEMINI_API_KEY"]:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def cost_usd(usage: dict, today: dt.date | None = None) -> float:
    """Conservative cost of one interaction from its `usage` block.

    The docs do not say whether total_output_tokens already includes thought
    tokens, or whether total_input_tokens includes tool-use tokens, so both are
    added separately. This can only overestimate, which is the safe direction
    for a budget.
    """
    p = prices(today)
    u = usage or {}
    cached = u.get("total_cached_tokens") or 0
    inp = max((u.get("total_input_tokens") or 0) - cached, 0)
    tool = u.get("total_tool_use_tokens") or 0
    out = (u.get("total_output_tokens") or 0) + (u.get("total_thought_tokens") or 0)
    return ((inp + tool) * p["input"] + cached * p["cached"] + out * p["output"]) / 1e6


@contextmanager
def _locked_ledger():
    HOME.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            yield f
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _entries(f):
    out = []
    for line in f.read().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def spend_summary(now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now()
    month, day = now.strftime("%Y-%m"), now.strftime("%Y-%m-%d")
    with _locked_ledger() as f:
        entries = _entries(f)
    m = [e for e in entries if e.get("ts", "").startswith(month)]
    d = [e for e in m if e.get("ts", "").startswith(day)]
    return {
        "month": month,
        "month_usd": round(sum(e.get("cost_usd", 0) for e in m), 4),
        "day_usd": round(sum(e.get("cost_usd", 0) for e in d), 4),
        "month_search_queries": sum(e.get("search_queries", 0) for e in m),
        "month_calls": len(m),
    }


class BudgetError(RuntimeError):
    pass


# Optional per-run cap (audit/background): all calls in this process share it.
import threading as _threading
_RUN = {"cap": None, "spent": 0.0}
_RUN_LOCK = _threading.Lock()


def set_run_cap(cap_usd: float | None):
    with _RUN_LOCK:
        _RUN["cap"], _RUN["spent"] = cap_usd, 0.0


def run_spent() -> float:
    return _RUN["spent"]


class RunCapReached(BudgetError):
    pass


def check_budget(effort: str, allow_over_daily: bool = False, expected_queries: int = 10,
                 reserve_usd: float | None = None):
    cfg = load_config()
    s = spend_summary()
    cap = float(cfg["MONTHLY_CAP_USD"])
    daily = float(cfg["DAILY_SOFT_CAP_USD"])
    qcap = int(cfg["MONTHLY_SEARCH_QUERY_CAP"])
    reserve = max(RESERVE_USD[effort], reserve_usd or 0)
    if _RUN["cap"] is not None and _RUN["spent"] + reserve > _RUN["cap"]:
        raise RunCapReached(
            f"run cap: this run spent ${_RUN['spent']:.3f} of its ${_RUN['cap']:.2f}; next call reserves ${reserve:.2f}.")
    if s["month_usd"] + reserve > cap:
        raise BudgetError(
            f"monthly cap: spent ${s['month_usd']:.2f} of ${cap:.2f} this month; "
            f"this call reserves ${reserve:.2f}. Refused."
        )
    if s["month_search_queries"] + expected_queries > qcap:
        raise BudgetError(
            f"search query cap: {s['month_search_queries']} of {qcap} used this month. Refused."
        )
    if s["day_usd"] + reserve > daily and not allow_over_daily:
        raise BudgetError(
            f"daily soft cap: spent ${s['day_usd']:.2f} of ${daily:.2f} today. "
            "Ask the user; rerun with --allow-over-daily only after they agree."
        )
    return s


def estimate_usd(input_chars: int, output_tokens: int = 15000) -> float:
    """Rough pre-call estimate for long inputs (about 3.5 chars per token), x1.5 margin."""
    p = prices()
    return 1.5 * (input_chars / 3.5 * p["input"] + output_tokens * p["output"]) / 1e6


def record_spend(entry: dict):
    with _RUN_LOCK:
        _RUN["spent"] += entry.get("cost_usd") or 0
    entry = {"ts": dt.datetime.now().isoformat(timespec="seconds"), **entry}
    with _locked_ledger() as f:
        f.seek(0, os.SEEK_END)
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def new_run_dir(mode: str, topic: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:40].strip("-") or "run"
    base = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{mode}-{slug}"
    for i in range(100):
        d = RUNS / (base if i == 0 else f"{base}-{i}")
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            continue
    raise RuntimeError("could not create a run directory")


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
