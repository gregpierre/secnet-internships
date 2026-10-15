import json

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
    monkeypatch.setattr(matcher, "JD_CACHE_PATH", str(tmp_path / "jd.json"))
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
    monkeypatch.setattr(matcher, "_send", lambda body: sent.append(body) or True)
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
    monkeypatch.setattr(matcher, "_send", lambda body: False)
    assert matcher.send_new_roles({"a": ROLE}, ["a"]) == []


def test_per_run_cap(env, monkeypatch):
    monkeypatch.setenv("MATCH_MAX_PER_RUN", "1")
    monkeypatch.setattr(matcher, "analyze", lambda *a: ANALYSIS)
    monkeypatch.setattr(matcher, "_send", lambda body: True)
    data = {"a": {**ROLE, "title": "A"}, "b": {**ROLE, "title": "B"}}
    assert len(matcher.send_new_roles(data, ["a", "b"])) == 1
