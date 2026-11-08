"""Tests for the per-user token bucket with an injected fake clock (no sleeping).

Each test runs its scenario inside a single `asyncio.run`, because asyncio.Lock binds to
the loop on first acquire and cannot be reused across separate loops.
"""

import asyncio

import pytest

from app.core.errors import RateLimitError
from app.core.rate_limit import RateLimiter, TokenBucket

REFILL_PER_S = 2.0


def _limiter(capacity: float = 3) -> tuple[RateLimiter, list[float]]:
    """A limiter whose clock always reads the last value appended to the returned list."""
    times: list[float] = []

    def clock() -> float:
        return times[-1] if times else 0.0

    limiter = RateLimiter(capacity, REFILL_PER_S, clock=clock)
    times.append(0.0)
    return limiter, times


def _burst(limiter: RateLimiter, times: list[float], user_id: str, count: int) -> None:
    """Perform `count` checks at the same instant inside one event loop."""

    async def scenario() -> None:
        for _ in range(count):
            await limiter.check(user_id)
            times.append(times[-1])  # same instant: the whole burst arrives together

    asyncio.run(scenario())


def test_fourth_request_is_limited_with_positive_retry_after():
    limiter, times = _limiter(capacity=3)
    _burst(limiter, times, "vera", 3)
    with pytest.raises(RateLimitError) as exc_info:
        _burst(limiter, times, "vera", 1)
    assert exc_info.value.retry_after_s > 0


def test_after_refill_delay_one_more_request_passes():
    limiter, times = _limiter(capacity=3)
    _burst(limiter, times, "vera", 3)
    with pytest.raises(RateLimitError):
        _burst(limiter, times, "vera", 1)
    times.append(times[-1] + 1 / REFILL_PER_S)  # exactly one token refilled
    _burst(limiter, times, "vera", 1)  # should not raise


def test_buckets_are_per_user():
    limiter, times = _limiter(capacity=3)
    _burst(limiter, times, "vera", 3)
    with pytest.raises(RateLimitError):
        _burst(limiter, times, "vera", 1)
    # vera is empty but anil still has a full bucket
    _burst(limiter, times, "anil", 3)


def test_tokens_never_exceed_capacity_after_long_idle():
    bucket = TokenBucket(capacity=3, refill_per_s=REFILL_PER_S)
    assert bucket.take(0.0) == 0.0
    assert bucket.tokens == 2.0
    # a long idle period must not refill past capacity
    assert bucket.take(1_000_000.0) == 0.0
    assert bucket.tokens == 2.0
