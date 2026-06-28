# Failure 5: The Auth Deadlock

> **Production symptom:** 40% of users received empty responses on supplier status queries.
> The other 60% saw normal results. The failures were intermittent, non-reproducible
> in testing, and produced no logged errors. The agent returned empty supplier data
> silently — not an error message, just missing results.

---

## What the Symptom Looked Like

```
User A: "What's the current delivery status for Apex Industries?"
Agent:  "Apex Industries has 3 active contracts with current delivery status: on-time
         for PO-2341, delayed 2 days for PO-2389, at-risk for PO-2401."

[Same query, 5 seconds later]

User B: "What's the current delivery status for Apex Industries?"
Agent:  "I wasn't able to retrieve the current delivery status for Apex Industries.
         The supplier data appears to be unavailable at this time."

[No error in logs. No exception. Just empty data.]
```

40% failure rate. Completely random-looking distribution.

---

## The Wrong Diagnosis

- "Supplier API is flaky" — SRE checked, API uptime was 99.98%
- "Rate limiting" — no 429s in logs
- "Network timeouts" — all requests completed, no timeouts
- "Some users have lower permissions" — same roles, same failure pattern

The 40% figure was a clue. It's very close to the percentage of concurrent users
in the system. But nobody connected those dots for a week.

---

## Actual Root Cause: OAuth2 Token Refresh Race Condition

```python
# BROKEN original oauth2.py — no mutex on token refresh

async def _get_valid_token(self) -> str:
    if self._token and not self._token.is_expired():
        return self._token.access_token

    # ← RACE CONDITION: multiple coroutines reach here simultaneously
    # when token expires during concurrent requests

    new_token = await self._refresh_token()   # All call this at once
    self._token = new_token                   # All overwrite with fresh tokens
    return self._token.access_token           # Some get invalidated tokens
```

**What happened:**

1. OAuth2 token expires during a busy period (concurrent users)
2. 5 coroutines all check: `token.is_expired()` → True
3. All 5 call `_refresh_token()` simultaneously
4. Token endpoint issues a new token on the **first** call
5. Tokens 2–5 get issued as well — but security policy **invalidates** them
   when a newer token is issued (rotate-on-use)
6. Coroutines 2–5 hold invalid tokens
7. Their API calls return 401 → exception caught → empty result returned
8. `last_error` was never propagated to the user — silent failure

**The 40% figure:** with 5 concurrent users, 4/5 = 80% would get invalid tokens.
But not all requests happened to hit the exact refresh window, so the observed rate
was 40% on average across the session. Pure statistics of a race condition.

---

## The Fix: asyncio.Lock() with double-checked locking

```python
# FIXED src/auth/oauth2.py

def __init__(self):
    self._token: OAuthToken | None = None
    self._refresh_lock = asyncio.Lock()  # Only one refresher at a time

async def _get_valid_token(self) -> str:
    # Fast path: no lock needed
    if self._token and not self._token.is_expired():
        return self._token.access_token

    # Slow path: acquire lock
    async with self._refresh_lock:
        # Double-checked: another coroutine may have refreshed while we waited
        if self._token and not self._token.is_expired():
            return self._token.access_token  # Already refreshed — use it

        # We hold the lock — safe to refresh
        await self._refresh_token()

    return self._token.access_token
```

**Why double-checked locking matters:**

- Coroutine A acquires lock, starts refreshing
- Coroutines B, C, D wait on the lock
- A finishes, releases lock
- B acquires lock — checks again — token is now valid — skips refresh
- C, D do the same
- Only 1 refresh call happens total

### Explicit error surfacing

The original code caught 401 exceptions and returned empty results.
The fix raises `SupplierAPIAuthError` explicitly, which the agent handles
with a user-visible error message instead of silent empty data.

---

## Before / After

| Metric | Before | After |
|--------|--------|-------|
| Supplier status failure rate | ~40% | < 0.1% |
| Concurrent token refreshes | Up to N | Always 1 |
| Error surfacing | Silent empty result | Explicit error message |
| Token refresh Prometheus metric | Not tracked | Tracked (success/failure) |

The Prometheus metric `procurement_agent_oauth_token_refresh_total{success="false"}`
went from ~4/minute to 0/minute after the fix.
