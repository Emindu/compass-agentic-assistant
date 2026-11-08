"""Per-user token-bucket rate limiting.

Why token bucket: it allows short bursts up to `capacity` (so a UI that fires a few calls
in quick succession still works) while enforcing a smooth long-run rate of
`refill_per_s`, and each check is O(1) per user with no scanning.

Trade-off: buckets live in process memory, which is correct only while the API runs as a
single replica. Production swaps this module for a Redis-backed bucket shared across
replicas; the `RateLimiter` interface and the FastAPI dependency stay the same.
"""

import asyncio
import time
from collections.abc import Callable
from functools import lru_cache

from fastapi import Depends

from app.api.auth import get_current_user
from app.core.config import get_settings
from app.core.errors import RateLimitError
from app.core.logging import get_logger
from app.security.rbac import UserContext

log = get_logger(__name__)


class TokenBucket:
    """One user's bucket: `tokens` refills at `refill_per_s` up to `capacity`.

    Returns are how the caller distinguishes outcomes without exceptions: 0.0 means a
    token was consumed and the request may proceed; a positive value is the seconds the
    caller should wait, with nothing consumed.
    """

    def __init__(self, capacity: float, refill_per_s: float) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.tokens = capacity  # start full so the first burst is allowed
        self.last = 0.0

    def take(self, now: float) -> float:
        """Refill from elapsed time, then consume one token if available."""
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_per_s)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        return (1 - self.tokens) / self.refill_per_s


class RateLimiter:
    """One bucket per user_id, guarded by an async lock.

    The lock matters because FastAPI serves request handlers concurrently; without it two
    requests could each observe `tokens >= 1` and both consume the last token. The clock
    is injectable so tests advance time instead of sleeping.
    """

    def __init__(
        self,
        capacity: float,
        refill_per_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.clock = clock
        self._lock = asyncio.Lock()
        self._buckets: dict[str, TokenBucket] = {}

    async def check(self, user_id: str) -> None:
        """Raise RateLimitError if the user's bucket is empty, else consume one token."""
        async with self._lock:
            bucket = self._buckets.setdefault(
                user_id, TokenBucket(self.capacity, self.refill_per_s)
            )
            wait_s = bucket.take(self.clock())
        if wait_s > 0:
            log.warning("rate_limited", user_id=user_id, retry_after_s=round(wait_s, 3))
            raise RateLimitError(retry_after_s=wait_s, detail="Rate limit exceeded")


@lru_cache
def get_rate_limiter() -> RateLimiter:
    """Shared limiter built from settings; lru_cache keeps state across requests in one process."""
    settings = get_settings()
    return RateLimiter(settings.rate_limit_capacity, settings.rate_limit_refill_per_s)


async def enforce_rate_limit(
    user: UserContext = Depends(get_current_user),
) -> UserContext:
    """Rate-limit after authentication so the bucket is keyed by a verified user_id.

    Returning the user lets endpoints chain `Depends(enforce_rate_limit)` alone and get
    both checks in one dependency, keeping route signatures small.
    """
    await get_rate_limiter().check(user.user_id)
    return user
