"""
BROKEN: OAuth2 client without mutex — causes Failure 5 (Auth Deadlock).

Demonstrates the race condition that caused 40% of users to get empty results.
Run this to see the race condition in action.
"""

import asyncio
import time
from dataclasses import dataclass


@dataclass
class FakeToken:
    value: str
    expires_at: float
    issued_at: float = 0

    def is_expired(self) -> bool:
        return time.time() >= self.expires_at


# Simulate token endpoint — rotate-on-use security model
_server_current_token = "token-INITIAL"
_server_issue_count = 0
_invalidated_tokens: set[str] = set()


async def _fake_token_endpoint() -> str:
    """
    Simulates an OAuth2 token endpoint with rotate-on-use policy.
    Each call issues a new token and invalidates the previous one.
    """
    global _server_current_token, _server_issue_count
    await asyncio.sleep(0.1)  # Network latency
    _server_issue_count += 1
    new_token = f"token-{_server_issue_count:04d}"
    if _server_current_token:
        _invalidated_tokens.add(_server_current_token)
    _server_current_token = new_token
    return new_token


async def _fake_api_call(token: str, user_id: str) -> dict | None:
    """API call that validates token server-side."""
    await asyncio.sleep(0.05)
    if token in _invalidated_tokens:
        return None  # 401 — invalid token, returns empty silently
    if token == _server_current_token:
        return {"supplier": "Apex Industries", "status": "on-time"}
    return None


# ─── BROKEN: No mutex on token refresh ───────────────────────────────────────

class BrokenOAuthClient:
    """
    BROKEN: Token refresh has no mutex.
    Multiple concurrent requests refresh simultaneously, invalidating each other's tokens.
    """

    def __init__(self):
        self._token: FakeToken | None = None

    async def _get_valid_token(self) -> str:
        if self._token and not self._token.is_expired():
            return self._token.value

        # BUG: No lock here. Multiple coroutines enter simultaneously.
        token_value = await _fake_token_endpoint()
        self._token = FakeToken(
            value=token_value,
            expires_at=time.time() + 0.5,   # Short TTL for demo
        )
        return self._token.value

    async def get_supplier_status(self, supplier_id: str, user_id: str) -> dict | None:
        token = await self._get_valid_token()
        result = await _fake_api_call(token, user_id)
        return result


async def demonstrate_race_condition():
    """
    Spawns 5 concurrent requests when the token is about to expire.
    Shows that ~80% get invalidated tokens and return None.
    """
    print("\nFAILURE 5 DEMONSTRATION — OAuth2 Token Refresh Race Condition")
    print("=" * 60)

    client = BrokenOAuthClient()

    # Get initial token (works fine)
    await client.get_supplier_status("SUP-001", "user-init")
    print(f"Initial token: {client._token.value}")

    # Force token expiry
    client._token.expires_at = time.time() - 1
    print(f"Token expired. Simulating 5 concurrent requests...")

    # 5 concurrent requests hit the expired token simultaneously
    results = await asyncio.gather(*[
        client.get_supplier_status("SUP-001", f"user-{i}")
        for i in range(5)
    ])

    print(f"\nResults (None = empty response, dict = success):")
    for i, result in enumerate(results):
        status = "SUCCESS" if result else "FAILED (silent empty result)"
        print(f"  User {i}: {status}")

    failed = sum(1 for r in results if r is None)
    print(f"\nFailed: {failed}/5 ({failed/5:.0%})")
    print(f"Tokens issued: {_server_issue_count}")
    print(f"Invalidated tokens: {_invalidated_tokens}")
    print()
    print("Fix: asyncio.Lock() in src/auth/oauth2.py — only 1 refresh ever happens")


if __name__ == "__main__":
    asyncio.run(demonstrate_race_condition())
