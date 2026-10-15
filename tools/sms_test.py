"""Send one AI match text right now, to prove the whole chain works.

    python tools/sms_test.py            # newest open role
    python tools/sms_test.py <job-id>   # a specific role

Ignores the score threshold and touches no queue or committed state, so it is
safe to run as often as you like (each run uses one Gemini call and one text).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from intern_engine import paths, sms, store  # noqa: E402


def main() -> None:
    missing = [k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
                           "TWILIO_TO_NUMBERS", "GEMINI_API_KEY", "RESUME_TEXT")
               if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing secrets: {', '.join(missing)}")

    data = store.load(paths.JOBS_PATH)
    if len(sys.argv) > 1:
        jid = sys.argv[1]
    else:
        open_roles = [(r.get("first_seen_at") or "", jid)
                      for jid, r in data.items() if r.get("is_open")]
        jid = max(open_roles)[1]
    record = data[jid]
    print(f"Analyzing: {record.get('company')} - {record.get('title')}")

    analysis = sms.analyze(record, sms._description(jid, record), sms._resume())
    # Only the score is logged: on a public repo the Actions log is world-readable,
    # and the full analysis quotes the resume.
    print(f"Score: {analysis['score']}/100")
    body = sms.format_message(record, analysis)
    print(f"Text: {len(body)} chars")
    sent = sms._send_sms(body)
    if not sent:
        sys.exit("Twilio did not accept the message (see error above).")
    print(f"Texted {sent} recipient(s).")


if __name__ == "__main__":
    main()
