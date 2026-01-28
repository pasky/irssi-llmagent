"""Tests for rate limiting functionality."""

import time

from muaddib.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test rate limiting behavior."""

    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        limiter = RateLimiter(rate=2, period=1)  # 2 requests per second

        # First two requests should pass
        assert limiter.check_limit() is True
        assert limiter.check_limit() is True

        # Third request should be blocked
        assert limiter.check_limit() is False

    def test_rate_limit_reset(self):
        """Test that rate limits reset after the period."""
        limiter = RateLimiter(rate=1, period=1)  # 1 request per second

        # First request passes
        assert limiter.check_limit() is True

        # Second request is blocked
        assert limiter.check_limit() is False

        # Wait for period to reset
        time.sleep(1.1)

        # Should be able to make requests again
        assert limiter.check_limit() is True

    def test_unlimited_rate(self):
        """Test that rate=0 means unlimited requests."""
        limiter = RateLimiter(rate=0, period=1)

        # Should always return True
        for _ in range(100):
            assert limiter.check_limit() is True

    def test_negative_rate(self):
        """Test that negative rate means unlimited requests."""
        limiter = RateLimiter(rate=-1, period=1)

        # Should always return True
        for _ in range(100):
            assert limiter.check_limit() is True
