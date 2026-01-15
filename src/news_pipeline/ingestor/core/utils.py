from __future__ import annotations

import functools
import hashlib
import logging
import random
import re
import threading
import time
from datetime import datetime
from typing import Callable, Optional, Type, Tuple, TypeVar
from urllib.parse import parse_qsl, urlparse, urlunparse, urlencode

try:  # pragma: no cover - optional dependency in tests
    from langdetect import detect  # type: ignore
except Exception:  # pragma: no cover
    detect = None

logger = logging.getLogger(__name__)

T = TypeVar("T")

STRIP_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}

PREFERRED_HTTPS = {
    "nytimes.com",
    "washingtonpost.com",
    "cnn.com",
    "foxnews.com",
    "bbc.com",
}


def canonicalize_url(raw: str) -> str:
    """Canonicalize URLs by normalizing host, scheme, query params, fragments, and AMP."""

    raw = raw.strip()
    if not raw:
        return ""

    u = urlparse(raw)
    scheme = (u.scheme or "http").lower()
    host = (u.hostname or "").lower()
    path = u.path or ""

    if host.startswith("www."):
        host = host[4:]

    path = re.sub(r"//+", "/", path)
    if path.endswith("/amp"):
        path = path[:-4]
    path = path.replace("/amp/", "/")

    if host in PREFERRED_HTTPS:
        scheme = "https"

    q = [(k, v) for k, v in parse_qsl(u.query, keep_blank_values=True) if k not in STRIP_KEYS]
    q.sort()

    canon = urlunparse((scheme, host, path.rstrip("/") or "/", "", urlencode(q), ""))
    return canon


_ws_re = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _ws_re.sub(" ", s)
    return s


def content_signature(title: str, lede: str) -> str:
    base = normalize_text(f"{title} {lede}")
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def compute_idempotency_key(url_canon: str, ts_pub: datetime, title: str) -> str:
    payload = f"{url_canon}|{ts_pub.isoformat()}|{normalize_text(title)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_language(text: str) -> Optional[str]:
    if not text:
        return None
    if detect is None:
        return None
    try:
        return detect(text)
    except Exception:
        return None


_INVALID_UNICODE_ESCAPE_RE = re.compile(r"\\u(?![0-9a-fA-F]{4})")


def sanitize_text(text: str) -> str:
    """Remove null bytes and other problematic Unicode characters for PostgreSQL."""
    if not isinstance(text, str):
        return text
    sanitized = text.replace('\x00', '').replace('\u0000', '')
    sanitized = _INVALID_UNICODE_ESCAPE_RE.sub(r"\\u", sanitized)
    return sanitized


def sanitize_json_data(data):
    """Recursively sanitize JSON data by removing null bytes from strings."""
    if isinstance(data, str):
        return sanitize_text(data)
    elif isinstance(data, dict):
        return {k: sanitize_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_data(item) for item in data]
    return data


# =============================================================================
# Retry with Exponential Backoff
# =============================================================================

# Default exceptions that indicate transient failures worth retrying
TRANSIENT_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# Try to add common HTTP/API exceptions if available
try:
    from openai import RateLimitError, APIConnectionError, APITimeoutError
    TRANSIENT_EXCEPTIONS = TRANSIENT_EXCEPTIONS + (RateLimitError, APIConnectionError, APITimeoutError)
except ImportError:
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = TRANSIENT_EXCEPTIONS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff on transient failures.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries, just run once)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff (delay = base_delay * exponential_base^attempt)
        jitter: If True, add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exception types that should trigger a retry

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_api():
            return client.chat.completions.create(...)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc
                    if attempt == max_retries:
                        logger.error(
                            "retry_exhausted func=%s attempts=%s error=%s",
                            func.__name__, attempt + 1, str(exc)[:200]
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        "retry_attempt func=%s attempt=%s/%s delay=%.2fs error=%s",
                        func.__name__, attempt + 1, max_retries + 1, delay, str(exc)[:100]
                    )
                    time.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry logic error in {func.__name__}")

        return wrapper
    return decorator


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

class CircuitBreaker:
    """
    Simple circuit breaker to prevent cascading failures when an API is down.

    States:
        CLOSED: Normal operation, requests go through
        OPEN: API is down, requests fail immediately without calling the API
        HALF_OPEN: Testing if API is back up, allowing one request through

    Usage:
        breaker = CircuitBreaker(name="openai", failure_threshold=5, recovery_timeout=60)

        @breaker
        def call_api():
            return client.chat.completions.create(...)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                # Check if we should transition to half-open
                if self._last_failure_time and (time.time() - self._last_failure_time) >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("circuit_breaker_half_open name=%s", self.name)
            return self._state

    def record_success(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                # Success in half-open state = close the circuit
                self._state = self.CLOSED
                logger.info("circuit_breaker_closed name=%s", self.name)
            self._failure_count = 0
            self._last_failure_time = None

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                # Failure in half-open state = open the circuit again
                self._state = self.OPEN
                logger.warning("circuit_breaker_reopened name=%s", self.name)
            elif self._failure_count >= self.failure_threshold:
                self._state = self.OPEN
                logger.error(
                    "circuit_breaker_opened name=%s failures=%s threshold=%s",
                    self.name, self._failure_count, self.failure_threshold
                )

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        current_state = self.state  # This may transition from OPEN to HALF_OPEN
        with self._lock:
            if current_state == self.CLOSED:
                return True
            elif current_state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as a decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not self.allow_request():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open - API calls blocked"
                )
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as exc:
                self.record_failure()
                raise
        return wrapper


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and blocking requests."""
    pass
