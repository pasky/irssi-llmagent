"""Rate limiting functionality."""

import time


class RateLimiter:
    """Simple time-window based rate limiter."""

    def __init__(self, rate: int = 30, period: int = 900):
        self.rate = rate
        self.period = period
        self.reset_time = 0
        self.count = 0

    def check_limit(self) -> bool:
        """Check if request is within rate limits."""
        if self.rate <= 0:
            return True

        now = time.time()
        if now >= self.reset_time:
            self.reset_time = now + self.period
            self.count = 0

        if self.count >= self.rate:
            return False

        self.count += 1
        return True
