"""Shared handling of Groq rate-limit responses.

Groq reports the exact wait in the 429 body, for example "try again in 8.6s" for a
per-minute budget or "try again in 7m33.6s" for the daily budget. The two need
different treatment: a per-minute wait is worth sleeping through, a daily wait is not.
"""

from __future__ import annotations

import re

# Groq: "Please try again in 8.6s" / "7m33.6s". Gemini: "Please retry in 31.27s".
_HINT = re.compile(
    r"(?:try again|retry) in\s*(?:(?P<h>\d+)h)?\s*(?:(?P<m>\d+)m(?!s))?"
    r"\s*(?:(?P<s>[0-9.]+)\s*(?P<unit>ms|s))?",
    re.IGNORECASE,
)


class BudgetExhaustedError(RuntimeError):
    """The provider asked for a wait so long that only a daily budget explains it."""


def parse_wait_hint(text: str) -> float | None:
    match = _HINT.search(text)
    if not match or not any(match.group(g) for g in ("h", "m", "s")):
        return None
    seconds = 0.0
    if match.group("h"):
        seconds += int(match.group("h")) * 3600
    if match.group("m"):
        seconds += int(match.group("m")) * 60
    if match.group("s"):
        value = float(match.group("s"))
        seconds += value / 1000 if (match.group("unit") or "s").lower() == "ms" else value
    return seconds


# Groq names the exhausted budget in words; Gemini names it in a quota id. Gemini uses
# the same "exceeded your current quota" sentence for a 15-request-per-minute limit and
# a 20-request-per-day limit, so only the quota id can tell the two apart.
_DAILY_MARKERS = ("per day", "(tpd)", "(rpd)", "perday")
_MINUTE_MARKERS = ("per minute", "(tpm)", "(rpm)", "(otpm)", "perminute")


def is_out_of_credit(exc: BaseException) -> bool:
    """The provider cannot serve this key at all right now, so waiting is pointless.

    HTTP 402 means depleted credit; HTTP 410 means the service is gone or browned out.
    Either way the right move is to fail over to the next model, not to retry.
    """
    status = getattr(exc, "status_code", None)
    if status in (402, 410):
        return True
    lowered = str(exc).lower()
    if "402" in lowered and ("credit" in lowered or "payment" in lowered):
        return True
    return "410" in lowered and ("retirement" in lowered or "brownout" in lowered)


def is_daily_limit(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _DAILY_MARKERS):
        return True
    if any(marker in lowered for marker in _MINUTE_MARKERS):
        return False
    # A quota message that names neither window and offers no wait is treated as
    # exhausted, since there is nothing sensible to wait for.
    return "exceeded your current quota" in lowered and parse_wait_hint(text) is None
