"""Unit tests for ``agent_study.async_client``.

Uses ``respx`` to stub HTTP responses and ``pytest-asyncio`` (in ``auto`` mode)
to run async coroutines directly. ``asyncio.sleep`` is monkey-patched in
retry tests so the suite stays fast and we can assert exact backoff values.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from time import time

import httpx
import pytest
import respx

from agent_study.async_client import (
    AsyncGitHubAPIError,
    AsyncGitHubClient,
    AsyncGitHubNotFoundError,
    AsyncGitHubRateLimitError,
    RateLimitInfo,
)


BASE_URL = "https://api.github.com"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _install_fake_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``asyncio.sleep`` inside the async_client module and record calls."""
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        "agent_study.async_client.asyncio.sleep", fake_sleep
    )
    return sleeps


# ---------------------------------------------------------------------------
# exception classes
# ---------------------------------------------------------------------------


class TestAsyncGitHubNotFoundError:
    def test_message_contains_status_and_path(self):
        err = AsyncGitHubNotFoundError("/repos/foo/bar")
        assert err.path == "/repos/foo/bar"
        msg = str(err)
        assert "404" in msg
        assert "/repos/foo/bar" in msg

    def test_is_subclass_of_api_error(self):
        err = AsyncGitHubNotFoundError("/x")
        assert isinstance(err, AsyncGitHubAPIError)


class TestAsyncGitHubRateLimitError:
    def test_attributes_stored(self):
        err = AsyncGitHubRateLimitError(
            status=429,
            path="/x",
            retry_after=2.0,
            reset_at=1_700_000_000,
            limit=60,
            remaining=0,
        )
        assert err.status == 429
        assert err.path == "/x"
        assert err.retry_after == 2.0
        assert err.reset_at == 1_700_000_000
        assert err.limit == 60
        assert err.remaining == 0
        assert "429" in str(err)

    def test_is_subclass_of_api_error(self):
        err = AsyncGitHubRateLimitError(429, "/x", None, None, 60, 0)
        assert isinstance(err, AsyncGitHubAPIError)

    def test_reset_in_returns_delta_when_future(self):
        future = int(time()) + 30
        err = AsyncGitHubRateLimitError(429, "/x", None, future, 60, 0)
        delta = err.reset_in
        assert delta is not None
        assert 25 <= delta <= 30

    def test_reset_in_none_when_reset_at_missing(self):
        err = AsyncGitHubRateLimitError(429, "/x", None, None, 60, 0)
        assert err.reset_in is None

    def test_reset_in_clamped_to_zero_when_past(self):
        past = int(time()) - 10
        err = AsyncGitHubRateLimitError(429, "/x", None, past, 60, 0)
        assert err.reset_in == 0.0


# ---------------------------------------------------------------------------
# RateLimitInfo
# ---------------------------------------------------------------------------


class TestRateLimitInfo:
    def test_from_headers_all_fields(self):
        headers = httpx.Headers(
            {
                "x-ratelimit-limit": "60",
                "x-ratelimit-remaining": "42",
                "x-ratelimit-reset": "1700000000",
                "x-ratelimit-used": "18",
            }
        )
        info = RateLimitInfo.from_headers(headers)
        assert info.limit == 60
        assert info.remaining == 42
        assert info.reset_at == 1_700_000_000
        assert info.used == 18

    def test_from_headers_missing_fields_are_none(self):
        info = RateLimitInfo.from_headers(httpx.Headers({}))
        assert info.limit is None
        assert info.remaining is None
        assert info.reset_at is None
        assert info.used is None

    def test_from_headers_invalid_int_becomes_none(self):
        headers = httpx.Headers({"x-ratelimit-limit": "not-a-number"})
        info = RateLimitInfo.from_headers(headers)
        assert info.limit is None

    def test_is_exhausted_true_when_remaining_zero(self):
        info = RateLimitInfo(limit=60, remaining=0, reset_at=None, used=60)
        assert info.is_exhausted is True

    def test_is_exhausted_false_when_remaining_positive(self):
        info = RateLimitInfo(limit=60, remaining=10, reset_at=None, used=50)
        assert info.is_exhausted is False

    def test_is_exhausted_false_when_remaining_none(self):
        info = RateLimitInfo(
            limit=None, remaining=None, reset_at=None, used=None
        )
        assert info.is_exhausted is False

    def test_is_frozen(self):
        info = RateLimitInfo(60, 0, None, 60)
        with pytest.raises(FrozenInstanceError):
            info.limit = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _compute_retry_delay (static method)
# ---------------------------------------------------------------------------


class TestComputeRetryDelay:
    def test_uses_retry_after_when_positive(self):
        exc = AsyncGitHubRateLimitError(
            429, "/x", retry_after=5.0, reset_at=None, limit=60, remaining=0
        )
        assert AsyncGitHubClient._compute_retry_delay(exc, attempt=0) == 5.0

    def test_ignores_retry_after_when_zero(self):
        exc = AsyncGitHubRateLimitError(
            429, "/x", retry_after=0.0, reset_at=None, limit=60, remaining=0
        )
        # Falls through to exponential backoff: 1.0
        assert AsyncGitHubClient._compute_retry_delay(exc, attempt=0) == 1.0

    def test_uses_reset_in_when_no_retry_after(self):
        future = int(time()) + 30
        exc = AsyncGitHubRateLimitError(
            429, "/x", retry_after=None, reset_at=future, limit=60, remaining=0
        )
        delay = AsyncGitHubClient._compute_retry_delay(exc, attempt=0)
        assert 25 <= delay <= 30

    def test_falls_back_to_exponential_backoff(self):
        exc = AsyncGitHubRateLimitError(
            429, "/x", retry_after=None, reset_at=None, limit=60, remaining=0
        )
        assert AsyncGitHubClient._compute_retry_delay(exc, attempt=0) == 1.0
        assert AsyncGitHubClient._compute_retry_delay(exc, attempt=1) == 2.0
        assert AsyncGitHubClient._compute_retry_delay(exc, attempt=2) == 4.0

    def test_caps_retry_after_at_max(self):
        exc = AsyncGitHubRateLimitError(
            429, "/x", retry_after=999.0, reset_at=None, limit=60, remaining=0
        )
        assert AsyncGitHubClient._compute_retry_delay(exc, attempt=0) == 60.0

    def test_caps_exponential_backoff_at_max(self):
        exc = AsyncGitHubRateLimitError(
            429, "/x", retry_after=None, reset_at=None, limit=60, remaining=0
        )
        # attempt 10: 1.0 * 2**10 = 1024 → capped at 60
        assert AsyncGitHubClient._compute_retry_delay(exc, attempt=10) == 60.0


# ---------------------------------------------------------------------------
# happy-path client behavior
# ---------------------------------------------------------------------------


class TestAsyncGitHubClientSuccess:
    @respx.mock
    async def test_get_json_returns_parsed_dict(self):
        respx.get(f"{BASE_URL}/ok").mock(
            return_value=httpx.Response(200, json={"hello": "world"})
        )
        async with AsyncGitHubClient() as client:
            result = await client.get_json("/ok")
        assert result == {"hello": "world"}

    @respx.mock
    async def test_last_rate_limit_populated_from_headers(self):
        respx.get(f"{BASE_URL}/ok").mock(
            return_value=httpx.Response(
                200,
                json={},
                headers={
                    "x-ratelimit-limit": "60",
                    "x-ratelimit-remaining": "55",
                    "x-ratelimit-reset": "1700000000",
                    "x-ratelimit-used": "5",
                },
            )
        )
        async with AsyncGitHubClient() as client:
            await client.get_json("/ok")
        info = client.last_rate_limit
        assert info is not None
        assert info.limit == 60
        assert info.remaining == 55
        assert info.reset_at == 1_700_000_000
        assert info.used == 5

    @respx.mock
    async def test_default_headers_accepted_and_user_agent(self):
        route = respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(200, json={})
        )
        async with AsyncGitHubClient() as client:
            await client.get_json("/x")
        request = route.calls[0].request
        assert request.headers["accept"] == "application/vnd.github+json"
        assert request.headers["user-agent"] == "agent-study"

    @respx.mock
    async def test_base_url_trailing_slash_stripped(self):
        route = respx.get(f"{BASE_URL}/users/octocat").mock(
            return_value=httpx.Response(200, json={})
        )
        async with AsyncGitHubClient(base_url=f"{BASE_URL}/") as client:
            await client.get_json("/users/octocat")
        assert route.called
        assert route.calls[0].request.url.path == "/users/octocat"

    async def test_aclose_marks_client_closed(self):
        client = AsyncGitHubClient()
        assert client.closed is False
        await client.aclose()
        assert client.closed is True

    async def test_context_manager_returns_client(self):
        async with AsyncGitHubClient() as client:
            assert isinstance(client, AsyncGitHubClient)
            assert not client.closed
        assert client.closed


# ---------------------------------------------------------------------------
# error-path client behavior
# ---------------------------------------------------------------------------


class TestAsyncGitHubClientErrors:
    @respx.mock
    async def test_404_raises_not_found_with_path(self):
        respx.get(f"{BASE_URL}/missing").mock(
            return_value=httpx.Response(404, json={"message": "Not Found"})
        )
        async with AsyncGitHubClient() as client:
            with pytest.raises(AsyncGitHubNotFoundError) as exc:
                await client.get_json("/missing")
        assert exc.value.path == "/missing"
        assert "404" in str(exc.value)

    @respx.mock
    async def test_429_raises_rate_limit_error_with_headers(self):
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(
                429,
                headers={
                    "retry-after": "2",
                    "x-ratelimit-limit": "60",
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": "1700000000",
                },
                json={"message": "rate limited"},
            )
        )
        async with AsyncGitHubClient() as client:
            with pytest.raises(AsyncGitHubRateLimitError) as exc:
                await client.get_json("/x")
        assert exc.value.status == 429
        assert exc.value.retry_after == 2.0
        assert exc.value.remaining == 0
        assert exc.value.limit == 60
        assert exc.value.reset_at == 1_700_000_000

    @respx.mock
    async def test_403_with_remaining_zero_raises_secondary_rate_limit(self):
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(
                403,
                headers={
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-limit": "60",
                    "x-ratelimit-reset": "1700000000",
                },
                json={"message": "secondary rate limit"},
            )
        )
        async with AsyncGitHubClient() as client:
            with pytest.raises(AsyncGitHubRateLimitError) as exc:
                await client.get_json("/x")
        assert exc.value.status == 403

    @respx.mock
    async def test_403_without_remaining_raises_generic_api_error(self):
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(403, json={"message": "forbidden"})
        )
        async with AsyncGitHubClient() as client:
            with pytest.raises(AsyncGitHubAPIError) as exc:
                await client.get_json("/x")
        assert not isinstance(exc.value, AsyncGitHubRateLimitError)
        assert "403" in str(exc.value)

    @respx.mock
    async def test_500_raises_generic_api_error(self):
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(500, json={"message": "boom"})
        )
        async with AsyncGitHubClient() as client:
            with pytest.raises(AsyncGitHubAPIError) as exc:
                await client.get_json("/x")
        assert "500" in str(exc.value)
        assert not isinstance(exc.value, AsyncGitHubRateLimitError)
        assert not isinstance(exc.value, AsyncGitHubNotFoundError)

    @respx.mock
    async def test_non_dict_json_body_raises_expected_object_error(self):
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(200, json=["not", "a", "dict"])
        )
        async with AsyncGitHubClient() as client:
            with pytest.raises(
                AsyncGitHubAPIError, match="Expected a JSON object"
            ):
                await client.get_json("/x")

    @respx.mock
    async def test_timeout_raises_api_error(self):
        respx.get(f"{BASE_URL}/x").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        async with AsyncGitHubClient() as client:
            with pytest.raises(AsyncGitHubAPIError, match="timed out"):
                await client.get_json("/x")

    @respx.mock
    async def test_network_error_raises_api_error(self):
        respx.get(f"{BASE_URL}/x").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        async with AsyncGitHubClient() as client:
            with pytest.raises(AsyncGitHubAPIError, match="network error"):
                await client.get_json("/x")

    async def test_request_after_close_raises_runtime_error(self):
        client = AsyncGitHubClient()
        await client.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await client.get_json("/x")

    @respx.mock
    async def test_last_rate_limit_still_updated_on_http_error(self):
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(
                429,
                headers={
                    "x-ratelimit-limit": "60",
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": "1700000000",
                    "x-ratelimit-used": "60",
                },
                json={"message": "rate limited"},
            )
        )
        async with AsyncGitHubClient() as client:
            with pytest.raises(AsyncGitHubRateLimitError):
                await client.get_json("/x")
        info = client.last_rate_limit
        assert info is not None
        assert info.remaining == 0
        assert info.used == 60


# ---------------------------------------------------------------------------
# retry behavior
# ---------------------------------------------------------------------------


class TestAsyncGitHubClientRetry:
    @respx.mock
    async def test_no_retry_when_max_retries_zero(self):
        route = respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(
                429,
                headers={"x-ratelimit-remaining": "0"},
                json={},
            )
        )
        async with AsyncGitHubClient(max_retries=0) as client:
            with pytest.raises(AsyncGitHubRateLimitError):
                await client.get_json("/x")
        assert route.call_count == 1

    @respx.mock
    async def test_retry_then_success(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _install_fake_sleep(monkeypatch)
        route = respx.get(f"{BASE_URL}/x").mock(
            side_effect=[
                httpx.Response(429, json={}),
                httpx.Response(429, json={}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        async with AsyncGitHubClient(max_retries=5) as client:
            result = await client.get_json("/x")
        assert result == {"ok": True}
        assert route.call_count == 3

    @respx.mock
    async def test_retry_exhausted_after_max_retries_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        sleeps = _install_fake_sleep(monkeypatch)
        route = respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(429, json={})
        )
        async with AsyncGitHubClient(max_retries=2) as client:
            with pytest.raises(AsyncGitHubRateLimitError):
                await client.get_json("/x")
        assert route.call_count == 3  # max_retries + 1
        assert len(sleeps) == 2  # one sleep between each pair of attempts

    @respx.mock
    async def test_per_call_max_retries_overrides_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _install_fake_sleep(monkeypatch)
        route = respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(429, json={})
        )
        async with AsyncGitHubClient(max_retries=5) as client:
            with pytest.raises(AsyncGitHubRateLimitError):
                await client.get_json("/x", max_retries=0)
        assert route.call_count == 1

    @respx.mock
    async def test_negative_instance_max_retries_clamped_to_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _install_fake_sleep(monkeypatch)
        route = respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(429, json={})
        )
        async with AsyncGitHubClient(max_retries=-3) as client:
            with pytest.raises(AsyncGitHubRateLimitError):
                await client.get_json("/x")
        assert route.call_count == 1
        assert client.max_retries == 0

    @respx.mock
    async def test_non_rate_limit_errors_are_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _install_fake_sleep(monkeypatch)
        route = respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(500, json={})
        )
        async with AsyncGitHubClient(max_retries=3) as client:
            with pytest.raises(AsyncGitHubAPIError):
                await client.get_json("/x")
        assert route.call_count == 1

    @respx.mock
    async def test_retry_uses_retry_after_header_when_positive(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        sleeps = _install_fake_sleep(monkeypatch)
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(
                429, headers={"retry-after": "5"}, json={}
            )
        )
        async with AsyncGitHubClient(max_retries=1) as client:
            with pytest.raises(AsyncGitHubRateLimitError):
                await client.get_json("/x")
        assert sleeps == [5.0]

    @respx.mock
    async def test_retry_uses_reset_in_when_retry_after_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        sleeps = _install_fake_sleep(monkeypatch)
        future = int(time()) + 30
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(
                429,
                headers={"x-ratelimit-reset": str(future)},
                json={},
            )
        )
        async with AsyncGitHubClient(max_retries=1) as client:
            with pytest.raises(AsyncGitHubRateLimitError):
                await client.get_json("/x")
        assert len(sleeps) == 1
        assert 25 <= sleeps[0] <= 30

    @respx.mock
    async def test_retry_falls_back_to_exponential_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        sleeps = _install_fake_sleep(monkeypatch)
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(429, headers={}, json={})
        )
        async with AsyncGitHubClient(max_retries=2) as client:
            with pytest.raises(AsyncGitHubRateLimitError):
                await client.get_json("/x")
        # attempt 0: 1.0, attempt 1: 2.0
        assert sleeps == [1.0, 2.0]

    @respx.mock
    async def test_retry_delay_capped_at_max(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        sleeps = _install_fake_sleep(monkeypatch)
        respx.get(f"{BASE_URL}/x").mock(
            return_value=httpx.Response(
                429, headers={"retry-after": "9999"}, json={}
            )
        )
        async with AsyncGitHubClient(max_retries=1) as client:
            with pytest.raises(AsyncGitHubRateLimitError):
                await client.get_json("/x")
        assert sleeps == [60.0]

    async def test_refuses_non_idempotent_method(self):
        client = AsyncGitHubClient()
        try:
            with pytest.raises(ValueError, match="non-idempotent"):
                await client._request_idempotent(
                    "POST", "/x", max_retries=None
                )
        finally:
            await client.aclose()

    async def test_method_is_uppercased_before_idempotency_check(self):
        client = AsyncGitHubClient()
        try:
            with pytest.raises(ValueError, match="non-idempotent"):
                await client._request_idempotent(
                    "delete", "/x", max_retries=None
                )
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


class TestAsyncGitHubClientProperties:
    def test_max_retries_property_reflects_construction_value(self):
        client = AsyncGitHubClient(max_retries=5)
        assert client.max_retries == 5

    async def test_closed_property_initially_false(self):
        async with AsyncGitHubClient() as client:
            assert client.closed is False

    async def test_closed_property_true_after_aclose(self):
        client = AsyncGitHubClient()
        await client.aclose()
        assert client.closed is True
