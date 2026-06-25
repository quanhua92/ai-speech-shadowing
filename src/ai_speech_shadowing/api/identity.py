"""Per-browser user identity via a signed cookie.

Each browser is assigned a random token (carried by the ``user_id`` cookie) on
first visit. The token itself is never written to disk — only its SHA-256
digest is used as the on-disk user identifier (the history subdirectory name).
This means a read of ``data/history/`` yields hashes that cannot be reversed
into valid cookies, so storage compromise alone cannot impersonate a user.

The cookie is ``httpOnly`` (XSS can't steal it via JS) and ``secure`` in
production (network can't sniff it). It is ``SameSite=Lax`` for CSRF defense
(the app's Origin-check middleware is the primary CSRF guard).
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid

USER_ID_COOKIE: str = "user_id"
"""The browser cookie name carrying the raw identity token."""

COOKIE_MAX_AGE: int = 365 * 24 * 3600
"""Cookie lifetime in seconds (1 year). The data itself is aged out by the
retention cleanup; this only controls how long the browser keeps the token."""

_USER_ID_RE: re.Pattern[str] = re.compile(r"^[a-f0-9]{64}$")
"""A valid on-disk user id: 64 lowercase hex chars (full sha256 digest)."""


def generate_token() -> str:
    """Generate a fresh identity token for a new browser (the cookie value).

    Returns the hex form of a random UUID4 (122 bits of entropy — unguessable).
    """
    return uuid.uuid4().hex


def hash_token(token: str) -> str:
    """Derive the on-disk user identifier from a raw token.

    The SHA-256 digest is used as the history subdirectory name. The raw token
    is never persisted; only this hash touches disk, so a disk read cannot be
    turned back into a valid cookie.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_valid_user_id(value: str | None) -> bool:
    """True when ``value`` is a well-formed on-disk user id (64 hex chars)."""
    return value is not None and bool(_USER_ID_RE.match(value))


def is_production() -> bool:
    """True when the app is running in production (cookies get ``secure``)."""
    return os.environ.get("ENV", "").lower() == "production"
