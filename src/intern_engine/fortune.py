"""Is this company on the Fortune 500? (Fortune 100 / 500 / 1000 / no)

data/fortune.json is Fortune's own 2026 ranking (the full Fortune 1000), name
-> rank. Refresh it each summer when the new list comes out:

    curl -s https://fortune.com/api/getRankingSearchYear/fortune500/<year>

Names on job boards rarely match Fortune's exactly ("Motorola" vs "Motorola
Solutions"), so a lookup tries, in order: the exact normalized name, a manual
alias, then the board name plus generic corporate words ("Cisco" -> "Cisco
Systems"). It never guesses the other way round: "Apple Bank" is not Apple and
"Nationwide Children's Hospital" is not Nationwide.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

from . import paths
from .priority import _key

FORTUNE_PATH = os.path.join(paths.DATA_DIR, "fortune.json")

# Words a board may leave off the Fortune name without changing the company.
_FILLER = {"holdings", "holding", "technologies", "technology", "systems",
           "laboratories", "international", "worldwide", "solutions", "group",
           "markets", "bancorp", "incorporated", "corp", "inc", "co"}

# Board name -> Fortune name, for the ones the rules above can't reach safely.
_ALIASES = {
    "booz allen": "Booz Allen Hamilton Holding",
    "air products": "Air Products & Chemicals",
    "elanco": "Elanco Animal Health",
    "fortune brands": "Fortune Brands Innovations",
    "mks instruments": "MKS",
    "general dynamics information technology": "General Dynamics",
    "gdit": "General Dynamics",
    "amazon web services": "Amazon",
    "aws": "Amazon",
    "google": "Alphabet",
    "meta": "Meta Platforms",
    "jpmorgan": "JPMorgan Chase",
    "jp morgan chase": "JPMorgan Chase",
}


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, int], dict[str, str], int]:
    try:
        with open(FORTUNE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}, {}, 0
    ranks = {_key(name): rank for name, rank in data.get("ranks", {}).items()}
    names = {_key(name): name for name in data.get("ranks", {})}
    return ranks, names, int(data.get("year") or 0)


def lookup(company: str) -> tuple[int, str] | None:
    """(rank, Fortune's name for it), or None if it isn't in the Fortune 1000."""
    ranks, names, _ = _index()
    key = _key(company)
    if not key:
        return None
    if key in ranks:
        return ranks[key], names[key]
    if key in _ALIASES:
        alias = _key(_ALIASES[key])
        if alias in ranks:
            return ranks[alias], names[alias]
    for fkey, rank in ranks.items():
        if fkey.startswith(key + " ") and set(fkey[len(key) + 1:].split()) <= _FILLER:
            return rank, names[fkey]
    return None


def year() -> int:
    return _index()[2]


def label(company: str) -> str:
    """'🏆 Fortune 100 · #16', '🏢 Fortune 500 · #378', '🏬 Fortune 1000 · #639'
    or 'Not in the Fortune 500'."""
    found = lookup(company)
    if not found:
        return "▫️ Not in the Fortune 500"
    rank = found[0]
    if rank <= 100:
        return f"🏆 <b>Fortune 100</b> · #{rank}"
    if rank <= 500:
        return f"🏢 <b>Fortune 500</b> · #{rank}"
    return f"🏬 Fortune 1000 · #{rank} (not top 500)"
