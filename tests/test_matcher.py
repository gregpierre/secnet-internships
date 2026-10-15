import json
from datetime import UTC, datetime

import pytest

from intern_engine import matcher

ROLE = {"company": "Acme Security", "title": "SOC Analyst Intern", "location": "Remote",
        "season": "Summer 2027", "url": "https://jobs.example.com/1", "is_open": True}

ANALYSIS = {
    "score": 78,
    "subscores": {"skills": 80, "projects": 75, "domain": 82, "level": 70},
    "verdict": "Apply — your triage work maps directly.",
    "strengths": [{"point": "Alert triage at Disney", "jd_quote": "triage alerts"}],
    "gaps": [{"point": "No Splunk SPL", "jd_quote": "experience with Splunk"}],
    "projects": [{"name": "Detection pack", "stack": "Sigma, Splunk", "what": "x",
                  "why": "y", "jd_quote": "write detections"}],
    "contacts": [
        {"name": "Jane Doe", "title": "SOC Analyst", "type": "intern->FT",
         "url": "https://www.linkedin.com/in/janedoe"},
        {"name": "Fake Person", "title": "x", "url": "https://example.com/nope"},
    ],
}

ENV = {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "1",
       "GEMINI_API_KEY": "k", "RESUME_TEXT": "resume"}


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MATCH_CATEGORIES", "*")
    monkeypatch.setenv("MATCH_TZ", "America/New_York")
    monkeypatch.setattr(matcher, "JD_CACHE_PATH", str(tmp_path / "jd.json"))
    monkeypatch.setattr(matcher, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(matcher, "_GEMINI_PAUSE", 0)
    monkeypatch.setattr(matcher.time, "sleep", lambda s: None)
    monkeypatch.setattr(matcher, "_description", lambda jid, r: "jd text")


def test_unconfigured_is_noop(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    assert matcher.send_new_roles({"a": ROLE}, ["a"]) == ["a"]


def test_message_drops_unverified_contacts():
    text = matcher.format_message(ROLE, ANALYSIS)
    assert "<b>78/100</b>" in text and "“experience with Splunk”" in text
    assert "linkedin.com/in/janedoe" in text and "Fake Person" not in text


def test_message_escapes_model_html():
    evil = {**ANALYSIS, "verdict": "<script>x</script> & more"}
    text = matcher.format_message({**ROLE, "company": "A&B <Co>"}, evil)
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "A&amp;B &lt;Co&gt;" in text


def test_every_line_has_balanced_tags():
    # split_message cuts between lines, so no tag may span two of them.
    for line in matcher.format_message(ROLE, ANALYSIS).split("\n"):
        for tag in ("b", "i", "code", "a"):
            assert line.count(f"<{tag}>") + line.count(f"<{tag} ") == line.count(f"</{tag}>")


def test_split_respects_telegram_cap():
    body = "\n".join(f"line {i} " + "x" * 90 for i in range(120))
    parts = matcher.split_message(body)
    assert len(parts) > 1 and all(len(p) <= 4096 for p in parts)
    assert parts[0].startswith("(1/")


def test_parse_tolerates_fences():
    out = matcher._parse("```json\n" + json.dumps(ANALYSIS) + "\n```")
    assert out["score"] == 78


def test_threshold_and_delivery(env, monkeypatch):
    scores = {"good": 80, "weak": 30}
    monkeypatch.setattr(matcher, "analyze",
                        lambda r, jd, res: {**ANALYSIS, "score": scores[r["title"]]})
    sent = []
    monkeypatch.setattr(matcher, "_send", lambda body, markup=None: sent.append(body) or True)
    data = {"g": {**ROLE, "title": "good"}, "w": {**ROLE, "title": "weak"},
            "c": {**ROLE, "is_open": False}}
    done = matcher.send_new_roles(data, ["g", "w", "c", "gone"])
    assert set(done) == {"g", "w", "c", "gone"}
    assert len(sent) == 1


def test_rate_limit_keeps_role_queued(env, monkeypatch):
    def boom(*a):
        raise matcher._Retryable("429")
    monkeypatch.setattr(matcher, "analyze", boom)
    assert matcher.send_new_roles({"a": ROLE}, ["a"]) == []


def test_failed_text_keeps_role_queued(env, monkeypatch):
    monkeypatch.setattr(matcher, "analyze", lambda *a: ANALYSIS)
    monkeypatch.setattr(matcher, "_send", lambda body, markup=None: False)
    assert matcher.send_new_roles({"a": ROLE}, ["a"]) == []


def test_per_run_cap(env, monkeypatch):
    monkeypatch.setenv("MATCH_MAX_PER_RUN", "1")
    monkeypatch.setattr(matcher, "analyze", lambda *a: ANALYSIS)
    monkeypatch.setattr(matcher, "_send", lambda body, markup=None: True)
    data = {"a": {**ROLE, "title": "A"}, "b": {**ROLE, "title": "B"}}
    assert len(matcher.send_new_roles(data, ["a", "b"])) == 1


# --- scope filter --------------------------------------------------------------

def test_relevant_uses_category_or_security_title(monkeypatch):
    monkeypatch.delenv("MATCH_CATEGORIES", raising=False)
    assert matcher.relevant({"category": "GRC / Risk", "title": "Intern"})
    assert matcher.relevant({"category": "Systems & Cloud Infra", "title": "Cyber Defense Intern"})
    assert not matcher.relevant({"category": "Systems & Cloud Infra",
                                 "title": "Presales Systems Engineer - 2027 Internship"})
    monkeypatch.setenv("MATCH_CATEGORIES", "*")
    assert matcher.relevant({"category": "Other", "title": "Barista"})


def test_out_of_scope_roles_settle_without_a_gemini_call(env, monkeypatch):
    monkeypatch.delenv("MATCH_CATEGORIES")
    called = []
    monkeypatch.setattr(matcher, "analyze", lambda *a: called.append(1) or ANALYSIS)
    data = {"x": {**ROLE, "title": "Presales Engineer", "category": "Other"}}
    assert matcher.send_new_roles(data, ["x"]) == ["x"]
    assert not called


# --- posted date ---------------------------------------------------------------

NOW = datetime(2026, 10, 15, 20, 0, tzinfo=UTC)   # 4pm Eastern


def test_posted_line_date_only_keeps_the_calendar_day(monkeypatch):
    monkeypatch.setenv("MATCH_TZ", "America/New_York")
    line = matcher.posted_line({"posted_at": "2026-10-13T00:00:00Z",
                                "posted_at_source": "date_only"}, NOW)
    assert line == "🕒 Posted Oct 13 · 2 days ago"


def test_posted_line_exact_and_fallback(monkeypatch):
    monkeypatch.setenv("MATCH_TZ", "America/New_York")
    assert matcher.posted_line({"posted_at": "2026-10-15T15:00:00Z",
                                "posted_at_source": "exact"}, NOW) == "🕒 Posted Oct 15 · 5h ago"
    assert matcher.posted_line({"first_seen_at": "2026-10-14T12:00:00Z"}, NOW) \
        == "🕒 Spotted Oct 14 · yesterday"
    assert matcher.posted_line({}, NOW) == ""


def test_message_flags_unread_posting_and_names_model():
    text = matcher.format_message(ROLE, {**ANALYSIS, "_has_jd": False, "_model": "gemini-x"})
    assert "Couldn't read the full posting" in text and "🤖 gemini-x" in text
    assert "Couldn't read" not in matcher.format_message(ROLE, {**ANALYSIS, "_has_jd": True})


# --- gemini consistency ----------------------------------------------------------

class _Resp:
    def __init__(self, status, text='{"score": 70}'):
        self.status_code, self.text = status, text

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self.text}]}}]}


def test_gemini_is_deterministic_and_retries_the_busy_main_model(env, monkeypatch):
    calls = []
    replies = iter([_Resp(503, "busy"), _Resp(200)])

    def post(url, json, **kw):
        calls.append((url.split("/models/")[1].split(":")[0], json["generationConfig"]))
        return next(replies)
    monkeypatch.setattr(matcher.httpx, "post", post)
    text, model = matcher._gemini("p")
    assert model == matcher._MODELS[0] and [c[0] for c in calls] == [model, model]
    assert calls[0][1]["temperature"] == 0 and calls[0][1]["responseMimeType"] == "application/json"


def test_gemini_out_of_quota_skips_straight_to_the_next_model(env, monkeypatch):
    replies = iter([_Resp(429, "quota"), _Resp(200)])
    seen = []
    monkeypatch.setattr(matcher.httpx, "post",
                        lambda url, json, **kw: seen.append(url) or next(replies))
    _, model = matcher._gemini("p")
    assert model == matcher._MODELS[1] and len(seen) == 2


# --- buttons, state, digest ----------------------------------------------------------

def test_scores_are_recorded_under_an_opaque_key(env, monkeypatch):
    monkeypatch.setattr(matcher, "analyze", lambda *a: ANALYSIS)
    markups = []
    monkeypatch.setattr(matcher, "_send", lambda body, markup=None: markups.append(markup) or True)
    matcher.send_new_roles({"job-123": ROLE}, ["job-123"])
    state = json.load(open(matcher.STATE_PATH))
    key = matcher.role_key("job-123")
    assert state["roles"][key]["s"] == 78 and "job-123" not in json.dumps(state)
    assert markups[0]["inline_keyboard"][0][0]["callback_data"] == f"m:a:{key}"


def _tap(update_id, data, chat="1"):
    return {"update_id": update_id, "callback_query": {
        "id": "q", "data": data, "message": {"chat": {"id": int(chat)}, "message_id": 5}}}


def test_sync_feedback_records_only_the_owners_taps(env, monkeypatch):
    calls = []

    def tg(method, payload):
        calls.append(method)
        if method == "getUpdates":
            return [_tap(10, "m:a:abc"), _tap(11, "m:x:def", chat="999"), _tap(12, "junk")]
        return True
    monkeypatch.setattr(matcher, "_tg", tg)
    assert matcher.sync_feedback(NOW) == 1
    state = json.load(open(matcher.STATE_PATH))
    assert state["roles"]["abc"]["st"] == "applied" and "def" not in state["roles"]
    assert state["tg_offset"] == 13 and "editMessageReplyMarkup" in calls


def test_digest_waits_for_the_hour_then_sends_once(env, monkeypatch):
    sent = []
    monkeypatch.setattr(matcher, "_send", lambda body, markup=None: sent.append(body) or True)
    good, done, low = matcher.role_key("g"), matcher.role_key("d"), matcher.role_key("l")
    t = "2026-10-15T18:00:00Z"
    json.dump({"roles": {good: {"s": 81, "t": t}, done: {"s": 90, "t": t, "st": "applied"},
                         low: {"s": 20, "t": t}}}, open(matcher.STATE_PATH, "w"))
    data = {"g": {**ROLE, "title": "Good"}, "d": {**ROLE, "title": "Done"},
            "l": {**ROLE, "title": "Low"}}
    assert matcher.send_digest(data, NOW) == 0 and not sent            # 4pm: not yet
    evening = datetime(2026, 10, 15, 23, 30, tzinfo=UTC)               # 7:30pm Eastern
    assert matcher.send_digest(data, evening) == 1
    assert "Good" in sent[0] and "Done" not in sent[0] and "Low" not in sent[0]
    assert matcher.send_digest(data, evening) == 0 and len(sent) == 1  # once a day
