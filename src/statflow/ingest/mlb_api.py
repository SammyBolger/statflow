"""Thin HTTP client for the MLB Stats API.

The client centralizes retry, timeout, and User-Agent handling so ingest
functions only need to think about which endpoint they're calling.
"""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from statflow.config import MLB_API_BASE_URL

# (connect, read) timeouts in seconds. Read is generous because some MLB
# endpoints (play-by-play for long games) return large payloads.
DEFAULT_TIMEOUT = (5, 30)

USER_AGENT = "statflow/0.1 (+https://github.com/SammyBolger/statflow)"


def _build_session() -> requests.Session:
    """Build a requests.Session with retry + reasonable defaults.

    Retries only GETs, only on transient errors (429 + 5xx). Backoff is
    exponential: 1s, 2s, 4s. Total of 3 retries — after that the caller
    sees the exception.
    """
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


class MLBStatsAPI:
    """Small wrapper around the MLB Stats API.

    Usage:
        api = MLBStatsAPI()
        data = api.get("schedule", params={"sportId": 1, "date": "2025-08-01"})
    """

    def __init__(
        self,
        base_url: str = MLB_API_BASE_URL,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else _build_session()

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET a JSON endpoint. Raises on any HTTP error after retries."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()
