"""AI resume-match texts for newly spotted roles (optional, best-effort).

For every new role this reads the job description, compares it against your
resume with Gemini, and — only when the match clears a threshold — texts you a
breakdown over Twilio:

  * a 1-100 match score with sub-scores (skills / projects / domain / level)
  * strengths and gaps, each backed by an exact quote from the posting
  * 2-3 projects worth building for THIS role: why each one matters, and the
    line of the posting it answers
  * 2-3 people at the company worth a LinkedIn connection request (former
    intern -> full-time, current full-time on the team, or a hiring manager)

Secrets (all required; any missing = silent no-op like every other channel):
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
  TWILIO_FROM_NUMBER   your Twilio number, e.g. +15551234567
  TWILIO_TO_NUMBERS    comma-separated destinations, e.g. +15557654321
  GEMINI_API_KEY       Google AI Studio key
  RESUME_TEXT          your resume as plain text (kept out of the public repo)

Optional tuning:
  SMS_MIN_SCORE        only text matches at or above this (default 50)
  SMS_MAX_PER_RUN      analyses per run; the rest wait for the next (default 6)
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
from urllib.parse import quote_plus

import httpx

from . import grouping, paths, sponsorship

_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_TWILIO = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest",
           "gemini-3.5-flash-lite")
_SMS_MAX = 1550          # Twilio's hard cap is 1600 per message; leave headroom
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


def _recipients() -> list[str]:
    return [n.strip() for n in _env("TWILIO_TO_NUMBERS").split(",") if n.strip()]


def configured() -> bool:
    return all(_env(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                                 "TWILIO_FROM_NUMBER", "GEMINI_API_KEY")) \
        and bool(_recipients()) and bool(_resume())


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
            print(f"  sms: {model} unavailable ({resp.status_code}), trying next model")
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

_ASCII = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "•": "*", "…": "...",
                        " ": " ", "→": "->"})


def _plain(text) -> str:
    """GSM-7 safe text. One curly quote flips a whole SMS to UCS-2, which
    cuts each segment from 153 to 67 characters and doubles the cost."""
    s = str(text or "").translate(_ASCII)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


def _q(text, limit: int = 140) -> str:
    s = _plain(text)
    return s if len(s) <= limit else s[:limit - 3].rstrip() + "..."


def _linkedin_search(company: str) -> str:
    q = quote_plus(f"{company} security")
    return f"https://www.linkedin.com/search/results/people/?keywords={q}"


def format_message(record: dict, a: dict) -> str:
    sub = a.get("subscores") or {}
    lines = [
        f"MATCH {a['score']}/100 - {_q(record.get('company'), 60)}",
        _q(record.get("title"), 120),
        " | ".join(_q(x, 60) for x in (record.get("location"), record.get("season")) if x),
        _plain(record.get("url")),
        "",
        "Skills {} | Projects {} | Domain {} | Level {}".format(
            sub.get("skills", "?"), sub.get("projects", "?"),
            sub.get("domain", "?"), sub.get("level", "?")),
    ]
    if a.get("verdict"):
        lines += [_q(a["verdict"], 220)]

    def section(title: str, items, mark: str) -> None:
        items = [i for i in (items or []) if isinstance(i, dict) and i.get("point")]
        if not items:
            return
        lines.extend(["", title])
        for i in items[:3]:
            lines.append(f"{mark} {_q(i['point'], 160)}")
            if i.get("jd_quote"):
                lines.append(f'  JD: "{_q(i["jd_quote"], 120)}"')

    section("STRENGTHS", a.get("strengths"), "+")
    section("GAPS", a.get("gaps"), "-")

    projects = [p for p in (a.get("projects") or []) if isinstance(p, dict) and p.get("name")]
    if projects:
        lines.extend(["", "BUILD THIS"])
        for n, p in enumerate(projects[:3], 1):
            head = f"{n}) {_q(p['name'], 80)}"
            if p.get("stack"):
                head += f" [{_q(p['stack'], 80)}]"
            lines.append(head)
            if p.get("what"):
                lines.append(f"  What: {_q(p['what'], 200)}")
            if p.get("why"):
                lines.append(f"  Why: {_q(p['why'], 180)}")
            if p.get("jd_quote"):
                lines.append(f'  JD: "{_q(p["jd_quote"], 120)}"')

    lines.extend(["", "CONNECT ON LINKEDIN"])
    contacts = [c for c in (a.get("contacts") or [])
                if isinstance(c, dict) and c.get("name")
                and "linkedin.com/in/" in str(c.get("url") or "")]
    for n, c in enumerate(contacts[:3], 1):
        who = f"{n}) {_q(c['name'], 50)}"
        if c.get("title"):
            who += f" - {_q(c['title'], 70)}"
        if c.get("type"):
            who += f" ({_q(c['type'], 25)})"
        lines += [who, f"  {_plain(c['url'])}"]
    if not contacts:
        lines.append("No verified profiles found. Search:")
    else:
        lines.append("(verify profiles before messaging) More:")
    lines.append(_linkedin_search(record.get("company") or ""))
    return "\n".join(lines)


def split_message(text: str, limit: int = _SMS_MAX) -> list[str]:
    """Split on line boundaries into Twilio-sized parts, labelled (1/3)..."""
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

def _send_sms(body: str) -> int:
    """Send to every recipient; returns how many received the full text."""
    sid, token = _env("TWILIO_ACCOUNT_SID"), _env("TWILIO_AUTH_TOKEN")
    url = _TWILIO.format(sid=sid)
    delivered = 0
    for to in _recipients():
        ok = True
        for part in split_message(body):
            try:
                httpx.post(url, auth=(sid, token), timeout=20, data={
                    "From": _env("TWILIO_FROM_NUMBER"), "To": to, "Body": part,
                }).raise_for_status()
            except Exception as exc:  # noqa: BLE001
                print(f"  sms: send to {to[-4:]} failed: {str(exc)[:200]}")
                ok = False
                break
            time.sleep(1)  # keep multi-part texts arriving in order
        delivered += ok
    return delivered


def send_new_roles(store_data: dict, new_ids: list[str]) -> list[str]:
    """Analyze and text new roles. Returns the ids that no longer need handling.

    Settled: unknown/closed roles, roles analyzed below the score threshold,
    roles whose analysis is permanently unparseable, and roles texted to at
    least one recipient. Held for next run: anything past the per-run cap and
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

    # Three requisitions for one job = one analysis, one text.
    grouped = [(row.get("id") or "", row)
               for row in grouping.group([{**r, "id": jid} for jid, r in records])]

    resume = _resume()
    threshold = _int_env("SMS_MIN_SCORE", 50)
    cap = _int_env("SMS_MAX_PER_RUN", 6)
    for n, (jid, record) in enumerate(grouped[:cap]):
        ids = [i for i in (record.get("opening_ids") or [jid]) if i]
        if n:
            time.sleep(_GEMINI_PAUSE)
        try:
            analysis = analyze(record, _description(jid, record), resume)
        except _Retryable as exc:
            print(f"  sms: Gemini unavailable, holding the rest: {str(exc)[:200]}")
            break
        except Exception as exc:  # noqa: BLE001 — permanent: don't loop forever
            print(f"  sms: skipped {record.get('company')}: {str(exc)[:200]}")
            settled += ids
            continue
        label = f"{record.get('company')} - {record.get('title')}"
        if analysis["score"] < threshold:
            print(f"  sms: {analysis['score']}/100 below {threshold}, not texted: {label}")
            settled += ids
            continue
        if _send_sms(format_message(record, analysis)):
            print(f"  sms: texted {analysis['score']}/100: {label}")
            settled += ids
        else:
            break  # Twilio down/misconfigured: keep it and the rest queued
    return settled
