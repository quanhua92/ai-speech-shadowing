"""Tests for the per-browser identity helpers (token, hash, cookie config)."""

from __future__ import annotations

import hashlib
import os
from unittest import mock

from ai_speech_shadowing.api.identity import (
    COOKIE_MAX_AGE,
    USER_ID_COOKIE,
    generate_token,
    hash_token,
    is_production,
    is_valid_user_id,
)


class TestTokenGeneration:
    def test_token_is_32_hex_chars(self) -> None:
        t = generate_token()
        assert len(t) == 32
        assert all(c in "0123456789abcdef" for c in t)

    def test_tokens_are_unique(self) -> None:
        tokens = {generate_token() for _ in range(100)}
        assert len(tokens) == 100


class TestHashToken:
    def test_hash_is_sha256_hex(self) -> None:
        t = generate_token()
        h = hash_token(t)
        assert h == hashlib.sha256(t.encode()).hexdigest()
        assert len(h) == 64

    def test_hash_is_deterministic(self) -> None:
        t = "abc123"
        assert hash_token(t) == hash_token(t)

    def test_different_tokens_hash_differently(self) -> None:
        assert hash_token("a") != hash_token("b")


class TestIsValidUserId:
    def test_64_hex_is_valid(self) -> None:
        assert is_valid_user_id("a" * 64)
        assert is_valid_user_id("0123456789abcdef" * 4)

    def test_short_string_invalid(self) -> None:
        assert not is_valid_user_id("abc")
        assert not is_valid_user_id("a" * 63)

    def test_non_hex_invalid(self) -> None:
        assert not is_valid_user_id("z" * 64)
        assert not is_valid_user_id("g" * 64)

    def test_none_invalid(self) -> None:
        assert not is_valid_user_id(None)


class TestIsProduction:
    def test_default_not_production(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENV", None)
            assert not is_production()

    def test_production_when_set(self) -> None:
        with mock.patch.dict(os.environ, {"ENV": "production"}):
            assert is_production()

    def test_case_insensitive(self) -> None:
        with mock.patch.dict(os.environ, {"ENV": "Production"}):
            assert is_production()

    def test_other_values_not_production(self) -> None:
        with mock.patch.dict(os.environ, {"ENV": "development"}):
            assert not is_production()


class TestConstants:
    def test_cookie_name(self) -> None:
        assert USER_ID_COOKIE == "user_id"

    def test_cookie_max_age_is_one_year(self) -> None:
        assert COOKIE_MAX_AGE == 365 * 24 * 3600
