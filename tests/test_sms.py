import json

import pytest

from intern_engine import sms

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
    monkeypatch.setattr(sms, "JD_CACHE_PATH", str(tmp_path / "jd.json"))
    monkeypatch.setattr(sms, "_GEMINI_PAUSE", 0)
    monkeypatch.setattr(sms.time, "sleep", lambda s: None)
    monkeypatch.setattr(sms, "_description", lambda jid, r: "jd text")


def test_unconfigured_is_noop(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    assert sms.send_new_roles({"a": ROLE}, ["a"]) == ["a"]


def test_message_is_ascii_and_drops_unverified_contacts():
    text = sms.format_message(ROLE, ANALYSIS)
    assert text.isascii()
    assert "MATCH 78/100" in text and 'JD: "experience with Splunk"' in text
    assert "linkedin.com/in/janedoe" in text and "Fake Person" not in text


def test_split_respects_telegram_cap():
    body = "\n".join(f"line {i} " + "x" * 90 for i in range(120))
    parts = sms.split_message(body)
    assert len(parts) > 1 and all(len(p) <= 4096 for p in parts)
    assert parts[0].startswith("(1/")


def test_parse_tolerates_fences():
    out = sms._parse("```json\n" + json.dumps(ANALYSIS) + "\n```")
    assert out["score"] == 78


def test_threshold_and_delivery(env, monkeypatch):
    scores = {"good": 80, "weak": 30}
    monkeypatch.setattr(sms, "analyze",
                        lambda r, jd, res: {**ANALYSIS, "score": scores[r["title"]]})
    sent = []
    monkeypatch.setattr(sms, "_send", lambda body: sent.append(body) or True)
    data = {"g": {**ROLE, "title": "good"}, "w": {**ROLE, "title": "weak"},
            "c": {**ROLE, "is_open": False}}
    done = sms.send_new_roles(data, ["g", "w", "c", "gone"])
    assert set(done) == {"g", "w", "c", "gone"}
    assert len(sent) == 1


def test_rate_limit_keeps_role_queued(env, monkeypatch):
    def boom(*a):
        raise sms._Retryable("429")
    monkeypatch.setattr(sms, "analyze", boom)
    assert sms.send_new_roles({"a": ROLE}, ["a"]) == []


def test_failed_text_keeps_role_queued(env, monkeypatch):
    monkeypatch.setattr(sms, "analyze", lambda *a: ANALYSIS)
    monkeypatch.setattr(sms, "_send", lambda body: False)
    assert sms.send_new_roles({"a": ROLE}, ["a"]) == []


def test_per_run_cap(env, monkeypatch):
    monkeypatch.setenv("SMS_MAX_PER_RUN", "1")
    monkeypatch.setattr(sms, "analyze", lambda *a: ANALYSIS)
    monkeypatch.setattr(sms, "_send", lambda body: True)
    data = {"a": {**ROLE, "title": "A"}, "b": {**ROLE, "title": "B"}}
    assert len(sms.send_new_roles(data, ["a", "b"])) == 1
