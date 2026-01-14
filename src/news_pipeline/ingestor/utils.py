from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Optional
from urllib.parse import parse_qsl, urlparse, urlunparse, urlencode

try:  # pragma: no cover - optional dependency in tests
    from langdetect import detect  # type: ignore
except Exception:  # pragma: no cover
    detect = None

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
