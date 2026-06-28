"""
Tests for Failure 5 fix: OAuth2 mutex token refresh.

Key assertions:
- Token is refreshed when expired
- Concurrent coroutines only trigger ONE refresh (mutex works)
- Double-checked locking avoids redundant refreshes
- Exponential backoff is applied on refresh failures
- SupplierAPIAuthError is raised on non-retriable failures (not silent None)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import pytest_asyncio
import httpx

from src.auth.oauth2 import OAuthToken, SupplierAPIClient, SupplierAPIAuthError


class TestOAuthToken:
    def test_not_expired_when_fresh(self):
        token = OAuthToken(access_token="tok", expires_at=time.time() + 3600)
        assert not token.is_expired()

    def test_expired_when_past_expiry(self):
        token = OAuthToken(access_token="tok", expires_at=time.time() - 1)
        assert token.is_expired()

    def test_buffer_seconds_triggers_early_expiry(self):
        """Token expiring in 20s should be considered expired with 30s buffer."""
        token = OAuthToken(access_token="tok", expires_at=time.time() + 20)
        assert token.is_expired(buffer_seconds=30)
        assert not token.is_expired(buffer_seconds=10)


class TestMutexTokenRefresh:
    """
    Core Failure 5 test: concurrent coroutines must trigger exactly ONE refresh.
    """

    @pytest.mark.asyncio
    async def test_concurrent_refresh_called_only_once(self):
        """
        Spawn N coroutines simultaneously against an expired token.
        The _refresh_token method must be called exactly once.
        """
        client = SupplierAPIClient()
        # Pre-set an expired token to guarantee all coroutines see expiry
        client._token = OAuthToken(access_token="old-token", expires_at=time.time() - 1)

        refresh_call_count = 0

        async def fake_refresh():
            nonlocal refresh_call_count
            refresh_call_count += 1
            await asyncio.sleep(0.05)  # Simulate network latency
            client._token = OAuthToken(
                access_token="new-token",
                expires_at=time.time() + 3600,
            )

        with patch.object(client, "_refresh_token", side_effect=fake_refresh):
            # Launch 10 concurrent callers
            tokens = await asyncio.gather(*[client._get_valid_token() for _ in range(10)])

        # All should get the same new token
        assert all(t == "new-token" for t in tokens)
        # But refresh was called exactly once (mutex worked)
        assert refresh_call_count == 1, (
            f"Expected 1 refresh call, got {refresh_call_count}. "
            "Mutex double-checked locking is broken."
        )

    @pytest.mark.asyncio
    async def test_double_checked_locking_skips_redundant_refresh(self):
        """
        Coroutine B waits on the lock while A refreshes.
        When B acquires the lock, it should see the fresh token and skip refresh.
        """
        client = SupplierAPIClient()
        client._token = OAuthToken(access_token="expired", expires_at=time.time() - 1)

        refresh_count = 0

        async def slow_refresh():
            nonlocal refresh_count
            refresh_count += 1
            await asyncio.sleep(0.1)
            client._token = OAuthToken(access_token="refreshed", expires_at=time.time() + 3600)

        with patch.object(client, "_refresh_token", side_effect=slow_refresh):
            results = await asyncio.gather(
                client._get_valid_token(),
                client._get_valid_token(),
            )

        assert refresh_count == 1
        assert all(r == "refreshed" for r in results)

    @pytest.mark.asyncio
    async def test_valid_token_skips_lock(self):
        """Fast path: valid token → no lock acquired, no refresh called."""
        client = SupplierAPIClient()
        client._token = OAuthToken(access_token="valid-tok", expires_at=time.time() + 3600)

        with patch.object(client, "_refresh_token", new_callable=AsyncMock) as mock_refresh:
            token = await client._get_valid_token()

        assert token == "valid-tok"
        mock_refresh.assert_not_called()


class TestExponentialBackoff:
    @pytest.mark.asyncio
    async def test_retries_with_backoff_on_server_error(self):
        """On 500 errors, client should retry up to MAX_REFRESH_RETRIES times."""
        client = SupplierAPIClient()
        client._token = OAuthToken(access_token="expired", expires_at=time.time() - 1)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        call_times = []

        async def fake_post(*args, **kwargs):
            call_times.append(time.time())
            raise httpx.HTTPStatusError("500", request=MagicMock(), response=mock_response)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_http = AsyncMock()
            mock_http.post = fake_post
            client._http = mock_http

            with pytest.raises(SupplierAPIAuthError):
                await client._refresh_token()

        # Should have slept between retries (exponential backoff)
        assert mock_sleep.call_count == client.MAX_REFRESH_RETRIES - 1

    @pytest.mark.asyncio
    async def test_401_raises_immediately_no_retry(self):
        """
        401/403 from token endpoint is non-retriable — raise immediately.
        Retrying against a bad client_secret is pointless.
        """
        client = SupplierAPIClient()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        async def fake_post(*args, **kwargs):
            raise httpx.HTTPStatusError("401", request=MagicMock(), response=mock_response)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_http = AsyncMock()
            mock_http.post = fake_post
            client._http = mock_http

            with pytest.raises(SupplierAPIAuthError, match="401"):
                await client._refresh_token()

        # Zero sleeps — raised immediately without backoff
        mock_sleep.assert_not_called()


class TestExplicitErrorSurfacing:
    """Failure 5: errors must be visible, not silently swallowed."""

    @pytest.mark.asyncio
    async def test_get_raises_supplier_api_error_on_401(self):
        """
        When a 401 comes back on an API call (not token refresh),
        raise SupplierAPIError explicitly — not return None silently.
        """
        from src.auth.oauth2 import SupplierAPIError
        client = SupplierAPIClient()
        client._token = OAuthToken(access_token="tok", expires_at=time.time() + 3600)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Token expired"

        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("401", request=MagicMock(), response=mock_response)
        )
        client._http = mock_http

        with pytest.raises(SupplierAPIError):
            await client.get("/suppliers/SUP-0001/status")

        # Token should be cleared so next call forces a refresh
        assert client._token is None
