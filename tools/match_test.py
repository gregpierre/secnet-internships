"""Send one AI match message to Telegram right now, to prove the whole chain works.

    python tools/match_test.py            # newest open in-scope role
    python tools/match_test.py <job-id>   # a specific role (skips the scope filter)
    python tools/match_test.py --digest   # today's top-matches digest, right now

Ignores the score threshold and the digest hour, and writes no committed state,
so it is safe to run as often as you like (each run uses one Gemini call and
one message). Buttons on a test message still work: the next scheduled run
picks up the tap.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from intern_engine import matcher, paths, store  # noqa: E402


def main() -> None:
    missing = [k for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                           "GEMINI_API_KEY", "RESUME_TEXT")
               if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing secrets: {', '.join(missing)}")

    data = store.load(paths.JOBS_PATH)
    args = [a for a in sys.argv[1:] if a]
    if args == ["--digest"]:
        listed = matcher.send_digest(data, force=True)
        print(f"Digest sent with {listed} role(s)." if listed
              else "Nothing scored in the last 24h yet; no digest sent.")
        return

    if args:
        jid = args[0]
    else:
        open_roles = [(r.get("first_seen_at") or "", jid)
                      for jid, r in data.items() if r.get("is_open") and matcher.relevant(r)]
        jid = max(open_roles)[1]
    record = data[jid]
    print(f"Analyzing: {record.get('company')} - {record.get('title')}")

    analysis = matcher.analyze(record, matcher._description(jid, record), matcher._resume())
    # Only the score is logged: on a public repo the Actions log is world-readable,
    # and the full analysis quotes the resume.
    print(f"Score: {analysis['score']}/100 via {analysis['_model']}"
          f"{'' if analysis['_has_jd'] else ' (posting not readable)'}")
    body = matcher.format_message(record, analysis)
    print(f"Text: {len(body)} chars")
    if not matcher._send(body, matcher._buttons(matcher.role_key(jid))):
        sys.exit("Telegram did not accept the message (see error above).")
    print("Sent to Telegram.")


if __name__ == "__main__":
    main()
