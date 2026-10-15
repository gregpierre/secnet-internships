"""AI resume-match messages for newly spotted roles (optional, best-effort).

For every new role this reads the job description, compares it against your
resume with Gemini, and — only when the match clears a threshold — sends you a
breakdown on Telegram (the same bot as telegram.py):

  * a 1-100 match score with sub-scores (skills / projects / domain / level)
  * strengths and gaps, each backed by an exact quote from the posting
  * 2-3 projects worth building for THIS role: why each one matters, and the
    line of the posting it answers
  * 2-3 people at the company worth a LinkedIn connection request (former
    intern -> full-time, current full-time on the team, or a hiring manager)

Secrets (all required; any missing = silent no-op like every other channel):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   shared with telegram.py
  GEMINI_API_KEY       Google AI Studio key
  RESUME_TEXT          your resume as plain text (kept out of the public repo)

Optional tuning:
  MATCH_MIN_SCORE      only send matches at or above this (default 50)
  MATCH_MAX_PER_RUN    analyses per run; the rest wait for the next (default 6)
  GEMINI_MODEL         model id (default gemini-3.8-flash, with fallbacks)
  MATCH_CATEGORIES     comma-separated store categories worth analyzing
                       (default: the security + cloud ones; "*" = all)
  MATCH_DIGEST_HOUR    local hour the daily digest goes out (default 19)
  MATCH_TZ             timezone for the digest and dates (default America/New_York)
  GEMINI_SEARCH        1 = Google Search grounding to find LinkedIn contacts
                       (off by default: the free tier rejects it with a 429)

Each message carries Applied / Not for me buttons. Telegram has no webhook
here, so taps are collected on the next run (sync_feedback) and the buttons
are then replaced with the choice. Once a day, after MATCH_DIGEST_HOUR local
time, send_digest posts a ranked list of the day's best untouched matches.

State lives in data/match_state.json, which is committed in a PUBLIC repo, so
roles are keyed by an HMAC of the job id (keyed with the bot token): the file
holds scores and choices but never says which company or role they belong to.

Same delivery contract as telegram.py: the return value is exactly the ids
that no longer need handling, so a rate limit or outage leaves roles queued
for the next run instead of dropping them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import unicodedata
from datetime import UTC, datetime, timedelta
from html import escape
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import httpx

from . import grouping, paths, sponsorship

_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_TELEGRAM = "https://api.telegram.org/bot{token}/{method}"
_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest",
           "gemini-3.5-flash-lite")
_MSG_MAX = 4000          # Telegram's hard cap is 4096 per message; leave headroom
_JD_MAX = 14000          # characters of posting text handed to the model
_GEMINI_PAUSE = 7        # seconds between analyses — free tier is ~10 req/min
_PRIMARY_RETRY = 15      # seconds to wait before retrying an overloaded main model
_MIN_JD = 300            # shorter than this is a shell page, not a posting
_DIGEST_MAX = 8
_STATE_KEEP_DAYS = 90
JD_CACHE_PATH = os.path.join(os.path.dirname(paths.OUTBOX_PATH), "jd_cache.json")
STATE_PATH = os.path.join(paths.DATA_DIR, "match_state.json")

# Store categories (pipeline classifier) worth a Gemini call by default. Plain
# "Systems & Cloud Infra" / IT support roles are mostly non-security and would
# burn the free tier's ~20 analyses a day; a security word in the title still
# lets one through.
_DEFAULT_CATEGORIES = ("Security (general)", "SOC / Detection", "GRC / Risk",
                       "AppSec / Product Sec", "Cloud & Infra Sec", "Cloud Platform")
_SECURITY_TITLE_RE = re.compile(
    r"secur|cyber|\bsoc\b|\bgrc\b|threat|vulnerab|detection|incident|"
    r"pen(etration)?[ -]?test|red team|blue team|forensic|\biam\b|identity", re.I)


class _Retryable(Exception):
    """Rate limit / outage: stop this run, keep the role queued."""


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _resume() -> str:
    text = _env("RESUME_TEXT")
    if not text and _env("RESUME_PATH") and os.path.exists(_env("RESUME_PATH")):
        with open(_env("RESUME_PATH"), encoding="utf-8") as f:
            text = f.read().strip()
    return text


def configured() -> bool:
    return all(_env(k) for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                                 "GEMINI_API_KEY")) and bool(_resume())


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _tz():
    try:
        return ZoneInfo(_env("MATCH_TZ") or "America/New_York")
    except Exception:  # noqa: BLE001 — bad name or no tzdata
        return UTC


def relevant(record: dict) -> bool:
    """Worth a Gemini call: an in-scope category, or a security word in the title."""
    wanted = _env("MATCH_CATEGORIES")
    if wanted == "*":
        return True
    cats = {c.strip() for c in wanted.split(",") if c.strip()} or set(_DEFAULT_CATEGORIES)
    return (record.get("category") in cats
            or bool(_SECURITY_TITLE_RE.search(record.get("title") or "")))


# --------------------------------------------------------------------------- #
# job description
# --------------------------------------------------------------------------- #

def cache_descriptions(jobs) -> None:
    """Called by the pipeline: keep the raw posting text for this run's roles.

    The store deliberately drops descriptions (multi-KB blobs), but `notify`
    runs in the same job right after `update`, so a local, uncommitted cache
    hands the analyzer the real text without re-fetching it.
    """
    try:
        cache: dict[str, str] = {}
        if os.path.exists(JD_CACHE_PATH):
            with open(JD_CACHE_PATH, encoding="utf-8") as f:
                cache = json.load(f) or {}
        for job in jobs:
            if getattr(job, "description", None):
                cache[job.id] = sponsorship.strip_html(job.description)[:_JD_MAX]
        # Bounded: only the most recent few hundred matter.
        cache = dict(list(cache.items())[-400:])
        with open(JD_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:  # noqa: BLE001 — a cache, never fatal
        pass


def _description(jid: str, record: dict) -> str:
    try:
        with open(JD_CACHE_PATH, encoding="utf-8") as f:
            cached = (json.load(f) or {}).get(jid)
        if cached and len(cached) >= _MIN_JD:
            return cached
    except Exception:  # noqa: BLE001
        pass
    url = record.get("url")
    if not url:
        return ""
    try:
        resp = httpx.get(url, timeout=20, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (secnet-internships)"})
        resp.raise_for_status()
        html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", resp.text)
        text = re.sub(r"\s+", " ", sponsorship.strip_html(html)).strip()
        # Client-rendered boards return a near-empty shell; treat as unread.
        return text[:_JD_MAX] if len(text) >= _MIN_JD else ""
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

_PROMPT = """You are a ruthless, specific cybersecurity recruiter and mentor. Compare the
candidate's resume to this internship and produce a brutally honest match analysis.

=== INTERNSHIP ===
Company: {company}
Title: {title}
Location: {location}
Cycle: {season}
Posting URL: {url}
Job description:
{jd}

=== CANDIDATE RESUME ===
{resume}

=== RULES ===
SCORING (integers 1-100, be calibrated, not generous):
- skills: overlap of required/preferred tools, languages and techniques.
- projects: how directly the candidate's projects/experience prove they can do THIS job's daily work.
- domain: fit with the team's area (SOC, GRC, cloud, AppSec, red team, network, IAM, etc.).
- level: eligibility/seniority fit (class year, degree timing, internship level).
- score: overall weighted match. 90+ = near-perfect, 70s = strong, 50s = plausible, <50 = weak.

STRENGTHS / GAPS: 3 each. Every item MUST include "jd_quote": a short EXACT quote
copied word-for-word from the job description that it relates to. Strengths name
the specific resume line that proves it. Gaps must be concrete (a named tool,
skill, or responsibility), never vague.

PROJECTS: 2-3 projects to build that would make this candidate stand out FOR THIS ROLE.
- NOT a generic homelab, NOT "set up a SIEM at home", NOT a certification, NOT a CTF.
- Must be unique, specific, and look like real work this team does: name the exact
  tools, data sources, and the concrete artifact produced (tool, detection pack,
  dashboard, report, pipeline).
- Must close a listed GAP or deepen a key requirement. Must not duplicate the
  candidate's existing projects (Cloud Security Posture Lab, Cloud Alert Triage &
  Runbook Validator) — building on them is fine.
- Buildable by one student in 1-3 weekends.
- "why": why a hiring manager for THIS team would care.
- "jd_quote": the exact posting line the project proves.

{contacts_rule}

=== OUTPUT ===
Return ONLY a JSON object, no markdown fences, no commentary:
{{
  "score": 0,
  "subscores": {{"skills": 0, "projects": 0, "domain": 0, "level": 0}},
  "verdict": "one sentence: should they apply and what's the angle",
  "strengths": [{{"point": "...", "jd_quote": "..."}}],
  "gaps": [{{"point": "...", "jd_quote": "..."}}],
  "projects": [{{"name": "...", "stack": "...", "what": "...", "why": "...", "jd_quote": "..."}}],
  "contacts": [{{"name": "...", "title": "...", "type": "intern->FT | full-time | hiring manager", "url": "https://www.linkedin.com/in/..."}}]
}}
Keep every string under 220 characters."""

_CONTACTS_SEARCH = """CONTACTS: Use Google Search to find 2-3 REAL people who work at {company} right now
and would be good to send a connection request about this role, in priority order:
(1) someone who interned there and converted to full-time in a similar team,
(2) a full-time employee on the relevant security team,
(3) a hiring manager / recruiter for security roles.
Only include people you actually found in search results with a linkedin.com/in/
URL and a current role at {company}. NEVER invent names or URLs. If you cannot
verify anyone, return an empty list."""

_CONTACTS_NONE = """CONTACTS: You have no web search, so you cannot verify anyone.
Return an empty list. NEVER invent names or URLs."""


def _search() -> bool:
    return _env("GEMINI_SEARCH").lower() in ("1", "true", "yes")


def _gemini(prompt: str) -> tuple[str, str]:
    """Returns (text, model that answered)."""
    key = _env("GEMINI_API_KEY")
    models = [_env("GEMINI_MODEL")] if _env("GEMINI_MODEL") else []
    models += [m for m in _MODELS if m not in models]
    # Temperature 0 + a fixed seed: the same posting should get the same score
    # run to run (at 0.4 one role drifted 65 -> 62 -> 55 across three runs).
    config = {"temperature": 0, "seed": 7, "maxOutputTokens": 4096}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config}
    if _search():
        # Search grounding is what lets it find real people at the company
        # (and read the posting when the board renders client-side).
        body["tools"] = [{"google_search": {}}]
    else:
        config["responseMimeType"] = "application/json"  # not allowed with tools
    attempts = [models[0], *models] if models else []  # main model gets a 2nd try
    last, busy, status = "", "", 0
    for n, model in enumerate(attempts):
        if n == 1 and attempts[0] == attempts[1]:
            if not busy or status == 429:
                continue  # gone or out of quota: waiting won't help
            print(f"  match: {model} busy, retrying in {_PRIMARY_RETRY}s")
            time.sleep(_PRIMARY_RETRY)
        try:
            resp = httpx.post(_GEMINI.format(model=model), json=body, timeout=120,
                              headers={"x-goog-api-key": key})
        except httpx.HTTPError as exc:
            busy, status = f"{model}: {exc}", 0
            continue
        status = resp.status_code
        if resp.status_code == 404:          # model id retired — try the next
            last = resp.text[:300]
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            # Overloaded or out of quota: each model has its own, so try the next.
            busy = f"gemini {resp.status_code} ({model}): {resp.text[:300]}"
            print(f"  match: {model} unavailable ({resp.status_code}), trying next model")
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"gemini {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        parts = (((data.get("candidates") or [{}])[0].get("content") or {})
                 .get("parts") or [])
        return "".join(p.get("text", "") for p in parts), model
    if busy:
        raise _Retryable(busy)
    raise RuntimeError(f"no Gemini model available: {last}")


def _parse(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict) or "score" not in data:
        raise ValueError("model output missing score")
    data["score"] = max(0, min(100, int(data.get("score") or 0)))
    return data


def analyze(record: dict, jd: str, resume: str) -> dict:
    prompt = _PROMPT.format(
        company=record.get("company") or "",
        title=record.get("title") or "",
        location=record.get("location") or "",
        season=record.get("season") or "",
        url=record.get("url") or "",
        jd=jd or ("(Description could not be fetched. Use Google Search to read the "
                  "posting at the URL above before scoring.)" if _search() else
                  "(Description could not be fetched. Score from the title and company "
                  "only, and say so in the verdict.)"),
        resume=resume,
        contacts_rule=(_CONTACTS_SEARCH if _search() else _CONTACTS_NONE).format(
            company=record.get("company") or ""),
    )
    error: Exception | None = None
    for _ in range(2):  # one retry for a malformed answer
        try:
            text, model = _gemini(prompt)
            data = _parse(text)
            data["_model"], data["_has_jd"] = model, bool(jd)
            return data
        except _Retryable:
            raise
        except (ValueError, json.JSONDecodeError) as exc:
            error = exc
    raise ValueError(f"unparseable analysis: {error}")


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
# Telegram HTML. Every tag opens and closes on one line, so split_message can
# cut between any two lines without breaking the markup.

def _clean(text) -> str:
    """Model output on one line: no stray control characters or runs of space."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch == " ")
    return re.sub(r"\s+", " ", s).strip()


def _q(text, limit: int = 140) -> str:
    """Clean, trimmed and HTML-escaped: safe to drop into the message."""
    s = _clean(text)
    if len(s) > limit:
        s = s[:limit - 1].rstrip() + "…"
    return escape(s, quote=False)


def _link(url, label: str) -> str:
    url = _clean(url)
    if not url.startswith(("http://", "https://")):
        return label
    return f'<a href="{escape(url)}">{label}</a>'


def _linkedin_search(company: str) -> str:
    q = quote_plus(f"{company} security")
    return f"https://www.linkedin.com/search/results/people/?keywords={q}"


def _bar(score: int) -> str:
    filled = max(0, min(10, round(score / 10)))
    return "▰" * filled + "▱" * (10 - filled)


def _badge(score: int) -> str:
    if score >= 75:
        return "🟢 Strong match"
    if score >= 60:
        return "🟡 Good match"
    if score >= 45:
        return "🟠 Stretch"
    return "🔴 Long shot"


def _sub(value) -> str:
    try:
        return str(max(0, min(100, int(value))))
    except (TypeError, ValueError):
        return "–"


def _parse_ts(value) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _ago(ts: datetime, now: datetime, date_only: bool) -> str:
    if date_only:
        days = (now.astimezone(_tz()).date() - ts.date()).days
        return "today" if days <= 0 else "yesterday" if days == 1 else f"{days} days ago"
    hours = int((now - ts).total_seconds() // 3600)
    if hours < 1:
        return "just now"
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return "yesterday" if days == 1 else f"{days} days ago"


def posted_line(record: dict, now: datetime | None = None) -> str:
    """'🕒 Posted Oct 13 · 2 days ago', falling back to when we first saw it."""
    now = now or datetime.now(UTC)
    ts = _parse_ts(record.get("posted_at"))
    label = "Posted"
    date_only = record.get("posted_at_source") == "date_only"
    if ts is None:
        ts, label, date_only = _parse_ts(record.get("first_seen_at")), "Spotted", False
    if ts is None:
        return ""
    # A date-only posting is a calendar day, not a UTC midnight: don't shift it.
    day = ts if date_only else ts.astimezone(_tz())
    return f"🕒 {label} {day.strftime('%b')} {day.day} · {_ago(ts, now, date_only)}"


def format_message(record: dict, a: dict) -> str:
    score = a["score"]
    sub = a.get("subscores") or {}
    where = " · ".join(f"{icon} {_q(x, 50)}" for icon, x in
                       (("📍", record.get("location")), ("🗓", record.get("season"))) if x)
    lines = [
        f"🎯 <b>{_q(record.get('company'), 60)}</b>",
        f"<b>{_q(record.get('title'), 120)}</b>",
    ]
    if where:
        lines.append(where)
    if posted_line(record):
        lines.append(posted_line(record))
    lines += [
        "",
        f"{_badge(score)} · <b>{score}/100</b>",
        f"<code>{_bar(score)}</code>",
        f"🛠 Skills <b>{_sub(sub.get('skills'))}</b> · 📂 Projects <b>{_sub(sub.get('projects'))}</b>",
        f"🧭 Domain <b>{_sub(sub.get('domain'))}</b> · 🎓 Level <b>{_sub(sub.get('level'))}</b>",
    ]
    if a.get("verdict"):
        lines += ["", f"💬 <i>{_q(a['verdict'], 240)}</i>"]
    if a.get("_has_jd") is False:
        lines += ["⚠️ <i>Couldn't read the full posting — scored from the title "
                  "and company only. Treat this score loosely.</i>"]

    def section(title: str, items) -> None:
        items = [i for i in (items or []) if isinstance(i, dict) and i.get("point")]
        if not items:
            return
        lines.extend(["", f"<b>{title}</b>"])
        for i in items[:3]:
            lines.append(f"• {_q(i['point'], 170)}")
            if i.get("jd_quote"):
                lines.append(f"   ↳ <i>“{_q(i['jd_quote'], 110)}”</i>")

    section("✅ Strengths", a.get("strengths"))
    section("⚠️ Gaps to close", a.get("gaps"))

    projects = [p for p in (a.get("projects") or []) if isinstance(p, dict) and p.get("name")]
    if projects:
        lines.extend(["", "<b>🚀 Projects to build</b>"])
        for n, p in enumerate(projects[:3], 1):
            lines.append(f"<b>{n}. {_q(p['name'], 80)}</b>")
            if p.get("stack"):
                lines.append(f"   🧰 <code>{_q(p['stack'], 90)}</code>")
            if p.get("what"):
                lines.append(f"   {_q(p['what'], 200)}")
            if p.get("why"):
                lines.append(f"   💡 {_q(p['why'], 180)}")

    lines.extend(["", "<b>🤝 People to connect with</b>"])
    contacts = [c for c in (a.get("contacts") or [])
                if isinstance(c, dict) and c.get("name")
                and "linkedin.com/in/" in str(c.get("url") or "")]
    for c in contacts[:3]:
        who = f"• {_link(c['url'], _q(c['name'], 50))}"
        if c.get("title"):
            who += f" · {_q(c['title'], 70)}"
        if c.get("type"):
            who += f" <i>({_q(c['type'], 25)})</i>"
        lines.append(who)
    company = record.get("company") or ""
    search = _link(_linkedin_search(company), f"Search {_q(company, 40)} security people")
    lines.append(f"🔎 {search}" if not contacts else f"🔎 Verify before messaging · {search}")

    lines.extend(["", f"👉 {_link(record.get('url'), '<b>Open the posting</b>')}"])
    if a.get("_model"):
        lines.append(f"<i>🤖 {_q(a['_model'], 40)}</i>")
    return "\n".join(lines)


def split_message(text: str, limit: int = _MSG_MAX) -> list[str]:
    """Split on line boundaries into Telegram-sized parts, labelled (1/3)..."""
    budget = limit - 8  # room for the "(n/m)\n" label
    parts: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > budget:          # pathological single line
            parts.append((cur + "\n" if cur else "") + line[:budget - len(cur) - 1])
            line, cur = line[budget - len(cur) - 1:], ""
        candidate = f"{cur}\n{line}" if cur else line
        if len(candidate) > budget:
            parts.append(cur)
            cur = line
        else:
            cur = candidate
    if cur:
        parts.append(cur)
    if len(parts) == 1:
        return parts
    return [f"({i}/{len(parts)})\n{p}" for i, p in enumerate(parts, 1)]


# --------------------------------------------------------------------------- #
# delivery
# --------------------------------------------------------------------------- #

def _tg(method: str, payload: dict) -> dict:
    """One Bot API call. Raises with Telegram's reason, never with the token."""
    token = _env("TELEGRAM_BOT_TOKEN")
    try:
        resp = httpx.post(_TELEGRAM.format(token=token, method=method),
                          json=payload, timeout=20)
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        # str(exc) can embed the request URL, which carries the bot token.
        raise RuntimeError(str(exc).replace(token, "***")[:300]) from None
    if not data.get("ok"):
        raise RuntimeError(f"telegram {resp.status_code}: {str(data)[:300]}")
    return data.get("result")


def _buttons(key: str) -> dict:
    return {"inline_keyboard": [[
        {"text": "✅ Applied", "callback_data": f"m:a:{key}"},
        {"text": "❌ Not for me", "callback_data": f"m:x:{key}"},
    ]]}


def _send(body: str, markup: dict | None = None) -> bool:
    """Post to the Telegram chat; True if every part arrived. The buttons ride
    on the last part, so they sit under the whole breakdown."""
    parts = split_message(body)
    for n, part in enumerate(parts, 1):
        payload = {"chat_id": _env("TELEGRAM_CHAT_ID"), "text": part,
                   "parse_mode": "HTML", "disable_web_page_preview": True}
        if markup and n == len(parts):
            payload["reply_markup"] = markup
        try:
            _tg("sendMessage", payload)
        except Exception as exc:  # noqa: BLE001
            print(f"  match: telegram send failed: {exc}")
            return False
        time.sleep(1)  # keep multi-part messages arriving in order
    return True


# --------------------------------------------------------------------------- #
# state (committed to a public repo — keyed by HMAC, see module docstring)
# --------------------------------------------------------------------------- #

def role_key(jid: str) -> str:
    secret = _env("TELEGRAM_BOT_TOKEN").encode()
    return hmac.new(secret, jid.encode(), hashlib.sha256).hexdigest()[:16]


def _load_state() -> dict:
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)
        if isinstance(state, dict) and isinstance(state.get("roles"), dict):
            return state
    except Exception:  # noqa: BLE001 — missing or damaged: start clean
        pass
    return {"roles": {}}


def _save_state(state: dict, now: datetime | None = None) -> None:
    cutoff = (now or datetime.now(UTC)) - timedelta(days=_STATE_KEEP_DAYS)
    state["roles"] = {k: v for k, v in state["roles"].items()
                      if (_parse_ts(v.get("t")) or cutoff) >= cutoff}
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, STATE_PATH)


def _record(state: dict, ids: list[str], score: int, now: datetime) -> None:
    for jid in ids:
        entry = state["roles"].setdefault(role_key(jid), {})
        entry.update({"s": score, "t": now.strftime("%Y-%m-%dT%H:%M:%SZ")})


_CHOICES = {"a": ("applied", "✅ Marked as applied"), "x": ("skip", "❌ Marked not for me")}


def sync_feedback(now: datetime | None = None) -> int:
    """Collect button taps since the last run. Returns how many were recorded.

    Telegram holds updates for 24h and the update workflow runs every 30 min,
    so nothing is lost between runs. Taps get their answer on the next run:
    the buttons are swapped for the choice so it's visible in the chat.
    """
    if not configured():
        return 0
    now = now or datetime.now(UTC)
    state = _load_state()
    try:
        updates = _tg("getUpdates", {"offset": int(state.get("tg_offset", 0)),
                                     "timeout": 0, "allowed_updates": ["callback_query"]})
    except Exception as exc:  # noqa: BLE001
        print(f"  match: could not read button taps: {exc}")
        return 0
    recorded = 0
    for update in updates or []:
        state["tg_offset"] = max(int(state.get("tg_offset", 0)), update["update_id"] + 1)
        query = update.get("callback_query") or {}
        msg = query.get("message") or {}
        if str((msg.get("chat") or {}).get("id")) != _env("TELEGRAM_CHAT_ID"):
            continue  # only the owner's chat can mark roles
        parts = str(query.get("data") or "").split(":")
        if len(parts) != 3 or parts[0] != "m" or parts[1] not in _CHOICES:
            continue
        status, label = _CHOICES[parts[1]]
        entry = state["roles"].setdefault(parts[2], {"t": now.strftime("%Y-%m-%dT%H:%M:%SZ")})
        entry["st"] = status
        recorded += 1
        for method, payload in (
            ("answerCallbackQuery", {"callback_query_id": query.get("id"), "text": label}),
            ("editMessageReplyMarkup", {
                "chat_id": msg["chat"]["id"], "message_id": msg.get("message_id"),
                "reply_markup": {"inline_keyboard": [[{"text": label, "callback_data": "m:done:-"}]]},
            }),
        ):
            try:
                _tg(method, payload)
            except Exception:  # noqa: BLE001 — cosmetic; the choice is already saved
                pass
    _save_state(state, now)
    return recorded


def send_digest(store_data: dict, now: datetime | None = None, force: bool = False) -> int:
    """Once a day after MATCH_DIGEST_HOUR: the day's best untouched matches.

    Returns how many roles were listed (0 = not due, or nothing worth sending).
    """
    if not configured():
        return 0
    now = now or datetime.now(UTC)
    local = now.astimezone(_tz())
    today = local.date().isoformat()
    state = _load_state()
    if not force and (local.hour < _int_env("MATCH_DIGEST_HOUR", 19)
                      or state.get("digest_date") == today):
        return 0

    threshold = _int_env("MATCH_MIN_SCORE", 50)
    since = now - timedelta(hours=24)
    rows, handled = [], set()
    for jid, record in store_data.items():
        entry = state["roles"].get(role_key(jid))
        if not entry or not record.get("is_open"):
            continue
        ident = (record.get("company"), record.get("title"))
        if entry.get("st"):
            # The buttons carry one requisition's key; the choice covers the job.
            handled.add(ident)
            continue
        ts = _parse_ts(entry.get("t"))
        if ts and ts >= since and int(entry.get("s") or 0) >= threshold:
            rows.append((int(entry["s"]), jid, record))
    # Grouped requisitions share a score: list each job once.
    seen, top = set(handled), []
    for score, jid, record in sorted(rows, key=lambda r: -r[0]):
        ident = (record.get("company"), record.get("title"))
        if ident not in seen:
            seen.add(ident)
            top.append((score, record))
    top = top[:_DIGEST_MAX]

    if top:
        medals = ["🥇", "🥈", "🥉"]
        lines = [f"🌙 <b>Today's top matches</b> · {local.strftime('%a %b')} {local.day}", ""]
        for n, (score, record) in enumerate(top):
            mark = medals[n] if n < len(medals) else f"{n + 1}."
            lines.append(f"{mark} <b>{score}</b> · {_link(record.get('url'), _q(record.get('company'), 40))}"
                         f" — {_q(record.get('title'), 80)}")
            if posted_line(record, now):
                lines.append(f"     {posted_line(record, now)}")
        lines += ["", "<i>Tap ✅ / ❌ on a role's message to drop it from here.</i>"]
        if not _send("\n".join(lines)):
            return 0
    if not force:
        state["digest_date"] = today
        _save_state(state, now)
    return len(top)


def send_new_roles(store_data: dict, new_ids: list[str]) -> list[str]:
    """Analyze and message new roles. Returns the ids that no longer need handling.

    Settled: unknown/closed roles, roles analyzed below the score threshold,
    roles whose analysis is permanently unparseable, and roles delivered to
    Telegram. Held for next run: anything past the per-run cap and
    anything hit by a rate limit or outage.
    """
    if not configured():
        return list(new_ids)
    if not new_ids:
        return []

    live = [(jid, store_data[jid]) for jid in new_ids if jid in store_data]
    settled = [jid for jid in new_ids if jid not in store_data]
    settled += [jid for jid, r in live if not r.get("is_open")]
    records = [(jid, r) for jid, r in live if r.get("is_open")]
    if not records:
        return settled

    # Skip roles outside the security/cloud scope before spending a Gemini call.
    keep = []
    for jid, r in records:
        if relevant(r):
            keep.append((jid, r))
        else:
            settled.append(jid)
    if len(keep) < len(records):
        print(f"  match: skipped {len(records) - len(keep)} out-of-scope role(s)")
    records = keep
    if not records:
        return settled

    # Three requisitions for one job = one analysis, one message.
    grouped = [(row.get("id") or "", row)
               for row in grouping.group([{**r, "id": jid} for jid, r in records])]

    resume = _resume()
    state = _load_state()
    now = datetime.now(UTC)
    threshold = _int_env("MATCH_MIN_SCORE", 50)
    cap = _int_env("MATCH_MAX_PER_RUN", 6)
    for n, (jid, record) in enumerate(grouped[:cap]):
        ids = [i for i in (record.get("opening_ids") or [jid]) if i]
        if n:
            time.sleep(_GEMINI_PAUSE)
        try:
            analysis = analyze(record, _description(jid, record), resume)
        except _Retryable as exc:
            print(f"  match: Gemini unavailable, holding the rest: {str(exc)[:200]}")
            break
        except Exception as exc:  # noqa: BLE001 — permanent: don't loop forever
            print(f"  match: skipped {record.get('company')}: {str(exc)[:200]}")
            settled += ids
            continue
        label = f"{record.get('company')} - {record.get('title')}"
        _record(state, ids, analysis["score"], now)
        if analysis["score"] < threshold:
            print(f"  match: {analysis['score']}/100 below {threshold}, not sent: {label}")
            settled += ids
            continue
        if _send(format_message(record, analysis), _buttons(role_key(jid))):
            print(f"  match: sent {analysis['score']}/100: {label}")
            settled += ids
        else:
            break  # Telegram down/misconfigured: keep it and the rest queued
    _save_state(state, now)
    return settled
