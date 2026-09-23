"""One cheap authenticated request per configured provider, run before an evaluation.

A rejected key otherwise surfaces as every golden row failing one by one after the chain
has fallen through its other models, which is slow and hides the real cause. Listing
models costs no tokens and distinguishes a bad key from a rate limit or an outage.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Literal

import httpx

from qc_copilot.config import get_settings
from qc_copilot.llm.providers import PROVIDER_BASE_URLS

Status = Literal["ok", "rejected", "rate_limited", "unreachable", "missing"]

# OpenRouter serves its model list without a key, so the key endpoint is checked instead.
AUTH_PATHS = {"openrouter": "auth/key"}
REJECTION_MARKERS = ("api key", "unauthorized", "invalid_api_key", "authentication")


@dataclass(frozen=True)
class ProviderCheck:
    provider: str
    status: Status
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "rate_limited")

    def line(self) -> str:
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.provider:<11}{self.status}{suffix}"


def _excerpt(body: str, limit: int = 160) -> str:
    text = " ".join(body.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def check_provider(
    provider: str, api_key: str, transport: httpx.BaseTransport | None = None
) -> ProviderCheck:
    if not api_key:
        return ProviderCheck(provider, "missing", "no key configured")
    base_url = PROVIDER_BASE_URLS[provider].rstrip("/") + "/"
    path = AUTH_PATHS.get(provider, "models")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(base_url=base_url, timeout=15.0, transport=transport) as client:
            response = client.get(path, headers=headers)
    except httpx.HTTPError as exc:
        return ProviderCheck(provider, "unreachable", type(exc).__name__)
    code = response.status_code
    body = _excerpt(response.text)
    if code == 200:
        return ProviderCheck(provider, "ok")
    if code == 429:
        return ProviderCheck(provider, "rate_limited", f"HTTP {code}: {body}")
    if code in (401, 403) or (code == 400 and any(m in body.lower() for m in REJECTION_MARKERS)):
        return ProviderCheck(provider, "rejected", f"HTTP {code}: {body}")
    return ProviderCheck(provider, "unreachable", f"HTTP {code}: {body}")


def run_preflight(
    api_keys: dict[str, str],
    required: list[str],
    transport: httpx.BaseTransport | None = None,
    out=sys.stdout,
) -> int:
    checks = [
        check_provider(name, key, transport)
        for name, key in api_keys.items()
        if key or name in required
    ]
    for check in checks:
        print(check.line(), file=out)
    blocking = [c for c in checks if c.provider in required and not c.usable]
    for check in blocking:
        print(
            f"ERROR: provider {check.provider} is required but {check.status}; "
            f"check the {check.provider.upper()} key in the environment.",
            file=sys.stderr,
        )
    return 1 if blocking else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--require",
        nargs="*",
        default=[],
        choices=sorted(PROVIDER_BASE_URLS),
        help="providers that must authenticate for the command to succeed",
    )
    args = parser.parse_args(argv)
    return run_preflight(get_settings().api_keys(), args.require)


if __name__ == "__main__":
    sys.exit(main())
