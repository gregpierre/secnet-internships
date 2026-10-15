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
  GEMINI_SEARCH        1 = Google Search grounding to find LinkedIn contacts
                       (off by default: the free tier rejects it with a 429)

Same delivery contract as telegram.py: the return value is exactly the ids
that no longer need handling, so a rate limit or outage leaves roles queued
for the next run instead of dropping them.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from html import escape
from urllib.parse import quote_plus

import httpx

from . import grouping, paths, sponsorship

_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_TELEGRAM = "https://api.telegram.org/bot{token}/sendMessage"
_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest",
           "gemini-3.5-flash-lite")
_MSG_MAX = 4000          # Telegram's hard cap is 4096 per message; leave headroom
_JD_MAX = 14000          # characters of posting text handed to the model
_GEMINI_PAUSE = 7        # seconds between analyses — free tier is ~10 req/min
JD_CACHE_PATH = os.path.join(os.path.dirname(paths.OUTBOX_PATH), "jd_cache.json")


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
        if cached and len(cached) > 300:
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
        return text[:_JD_MAX]
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


def _gemini(prompt: str) -> str:
    key = _env("GEMINI_API_KEY")
    models = [_env("GEMINI_MODEL")] if _env("GEMINI_MODEL") else []
    models += [m for m in _MODELS if m not in models]
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 4096},
    }
    if _search():
        # Search grounding is what lets it find real people at the company
        # (and read the posting when the board renders client-side).
        body["tools"] = [{"google_search": {}}]
    last, busy = "", ""
    for model in models:
        try:
            resp = httpx.post(_GEMINI.format(model=model), json=body, timeout=120,
                              headers={"x-goog-api-key": key})
        except httpx.HTTPError as exc:
            busy = f"{model}: {exc}"
            continue
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
        return "".join(p.get("text", "") for p in parts)
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
            return _parse(_gemini(prompt))
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
    lines += [
        "",
        f"{_badge(score)} · <b>{score}/100</b>",
        f"<code>{_bar(score)}</code>",
        f"🛠 Skills <b>{_sub(sub.get('skills'))}</b> · 📂 Projects <b>{_sub(sub.get('projects'))}</b>",
        f"🧭 Domain <b>{_sub(sub.get('domain'))}</b> · 🎓 Level <b>{_sub(sub.get('level'))}</b>",
    ]
    if a.get("verdict"):
        lines += ["", f"💬 <i>{_q(a['verdict'], 240)}</i>"]

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

def _send(body: str) -> bool:
    """Post the breakdown to the Telegram chat; True if every part arrived."""
    url = _TELEGRAM.format(token=_env("TELEGRAM_BOT_TOKEN"))
    for part in split_message(body):
        try:
            resp = httpx.post(url, timeout=20, json={
                "chat_id": _env("TELEGRAM_CHAT_ID"), "text": part,
                "parse_mode": "HTML", "disable_web_page_preview": True,
            })
            if resp.status_code >= 400:
                # Telegram's JSON names the cause ("chat not found", ...).
                raise RuntimeError(f"telegram {resp.status_code}: {resp.text[:300]}")
        except Exception as exc:  # noqa: BLE001
            # str(exc) can embed the request URL, which carries the bot token.
            print(f"  match: telegram send failed: "
                  f"{str(exc).replace(_env('TELEGRAM_BOT_TOKEN'), '***')[:300]}")
            return False
        time.sleep(1)  # keep multi-part messages arriving in order
    return True


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

    # Three requisitions for one job = one analysis, one message.
    grouped = [(row.get("id") or "", row)
               for row in grouping.group([{**r, "id": jid} for jid, r in records])]

    resume = _resume()
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
        if analysis["score"] < threshold:
            print(f"  match: {analysis['score']}/100 below {threshold}, not sent: {label}")
            settled += ids
            continue
        if _send(format_message(record, analysis)):
            print(f"  match: sent {analysis['score']}/100: {label}")
            settled += ids
        else:
            break  # Telegram down/misconfigured: keep it and the rest queued
    return settled
