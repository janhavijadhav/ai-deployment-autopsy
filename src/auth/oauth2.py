"""
OAuth2 client with mutex-protected token refresh — the fix for Failure 5 (Auth Deadlock).

THE PROBLEM:
  The procurement agent runs multiple concurrent tool calls. When an OAuth2 token
  expires, all in-flight requests detect it simultaneously and all attempt to refresh.
  The token endpoint issues a new token on the first call and invalidates it on the
  second (rotate-on-use security). 40% of requests then fail with 401 (expired token)
  because they got the old token back from a race-condition refresh.

  The failures were SILENT — the agent returned empty results instead of an error,
  so users saw "No suppliers found" for valid queries. Took 2 weeks to trace.

THE FIX:
  1. asyncio.Lock() ensures only one coroutine refreshes at a time.
  2. After acquiring the lock, re-check if the token is STILL expired (double-checked
     locking) — if another coroutine already refreshed it, skip the refresh.
  3. Exponential backoff on refresh failures.
  4. Explicit error surfacing: auth failures raise SupplierAPIAuthError, not silent None.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx

from src.config import settings
from src.observability.tracing import tracer, metrics


# ─── Exceptions ──────────────────────────────────────────────────────────────

class SupplierAPIAuthError(Exception):
    """Raised when authentication cannot be established after retries."""


class SupplierAPIError(Exception):
    """Raised for non-auth API errors."""


# ─── Token storage ────────────────────────────────────────────────────────────

@dataclass
class OAuthToken:
    access_token: str
    expires_at: float        # unix timestamp
    token_type: str = "Bearer"

    def is_expired(self, buffer_seconds: float = 30.0) -> bool:
        """True if token expires within the next buffer_seconds."""
        return time.time() >= (self.expires_at - buffer_seconds)


# ─── Client ───────────────────────────────────────────────────────────────────

class SupplierAPIClient:
    """
    Async HTTP client for the Supplier REST API.
    Handles OAuth2 token lifecycle with mutex-protected refresh.
    """

    MAX_REFRESH_RETRIES = 3
    BACKOFF_BASE = 1.0          # seconds

    def __init__(self):
        self._token: OAuthToken | None = None
        self._refresh_lock = asyncio.Lock()   # THE KEY FIX — one refresher at a time
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=settings.SUPPLIER_API_BASE_URL,
                timeout=10.0,
            )
        return self._http

    async def _get_valid_token(self) -> str:
        """
        Return a valid access token, refreshing if needed.
        Uses double-checked locking to avoid the refresh race condition.
        """
        # Fast path: token is valid, no lock needed
        if self._token and not self._token.is_expired():
            return self._token.access_token

        # Slow path: token expired or missing — acquire lock
        async with self._refresh_lock:
            # Double-checked: another coroutine may have refreshed while we waited
            if self._token and not self._token.is_expired():
                return self._token.access_token

            # We hold the lock — safe to refresh
            with tracer.span("oauth2.token_refresh") as span:
                await self._refresh_token()
                span.set_attribute("token_acquired", True)

        return self._token.access_token

    async def _refresh_token(self) -> None:
        """
        Fetch a new token from the OAuth2 token endpoint.
        Retries with exponential backoff on failure.
        Raises SupplierAPIAuthError if all retries are exhausted.
        """
        http = await self._get_http()
        last_error: Exception | None = None

        for attempt in range(self.MAX_REFRESH_RETRIES):
            try:
                response = await http.post(
                    settings.OAUTH2_TOKEN_URL,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": settings.OAUTH2_CLIENT_ID,
                        "client_secret": settings.OAUTH2_CLIENT_SECRET,
                        "scope": settings.OAUTH2_SCOPE,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                data = response.json()

                self._token = OAuthToken(
                    access_token=data["access_token"],
                    expires_at=time.time() + data.get("expires_in", 3600),
                    token_type=data.get("token_type", "Bearer"),
                )
                metrics.record_token_refresh(success=True)
                return

            except httpx.HTTPStatusError as e:
                last_error = e
                metrics.record_token_refresh(success=False)
                if e.response.status_code in (400, 401, 403):
                    # Non-retriable auth errors
                    raise SupplierAPIAuthError(
                        f"OAuth2 token refresh rejected ({e.response.status_code}): {e.response.text}"
                    ) from e

            except Exception as e:
                last_error = e
                metrics.record_token_refresh(success=False)

            # Exponential backoff
            wait = self.BACKOFF_BASE * (2 ** attempt)
            await asyncio.sleep(wait)

        raise SupplierAPIAuthError(
            f"Token refresh failed after {self.MAX_REFRESH_RETRIES} attempts"
        ) from last_error

    async def get(self, path: str, **kwargs) -> dict:
        """Authenticated GET. Raises SupplierAPIError on HTTP errors."""
        token = await self._get_valid_token()
        http = await self._get_http()

        with tracer.span(f"supplier_api.GET {path}") as span:
            try:
                response = await http.get(
                    path,
                    headers={"Authorization": f"Bearer {token}"},
                    **kwargs,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    # Token was valid when we checked but rejected — force refresh on next call
                    self._token = None
                span.set_attribute("error", str(e))
                raise SupplierAPIError(
                    f"Supplier API {path} returned {e.response.status_code}"
                ) from e

    async def post(self, path: str, json: dict, **kwargs) -> dict:
        """Authenticated POST."""
        token = await self._get_valid_token()
        http = await self._get_http()

        with tracer.span(f"supplier_api.POST {path}"):
            response = await http.post(
                path,
                json=json,
                headers={"Authorization": f"Bearer {token}"},
                **kwargs,
            )
            response.raise_for_status()
            return response.json()

    async def aclose(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()
