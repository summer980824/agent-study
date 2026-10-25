"""``agent_study.llm_client`` 的单元测试。

用 ``respx`` 给 ``httpx`` 打桩，只验证请求契约与错误处理，不联网。
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from agent_study.llm_client import chat

BASE_URL = "https://api.example.com/v1"


def _ok_response(content: str = "三清山今天晴。") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        },
    )


def _body(route: respx.Route) -> dict:
    return json.loads(route.calls[0].request.content)


# ---------------------------------------------------------------------------
# 请求契约
# ---------------------------------------------------------------------------


class TestChatRequest:
    @respx.mock
    def test_posts_to_chat_completions_endpoint(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert route.calls[0].request.url.path == "/v1/chat/completions"

    @respx.mock
    def test_sends_bearer_authorization_header(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=BASE_URL, api_key="sk-secret", model="m")
        assert route.calls[0].request.headers["authorization"] == "Bearer sk-secret"

    @respx.mock
    def test_sends_model_and_messages(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("今天三清山天气如何？", base_url=BASE_URL, api_key="k", model="m")
        body = _body(route)
        assert body["model"] == "m"
        assert body["messages"] == [
            {"role": "user", "content": "今天三清山天气如何？"}
        ]

    @respx.mock
    def test_omits_optional_params_by_default(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        body = _body(route)
        assert "temperature" not in body
        assert "max_completion_tokens" not in body
        assert "reasoning_effort" not in body

    @respx.mock
    def test_sends_reasoning_effort_when_given(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=BASE_URL, api_key="k", model="m", reasoning_effort="low")
        assert _body(route)["reasoning_effort"] == "low"

    @respx.mock
    def test_sends_temperature_when_given(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=BASE_URL, api_key="k", model="m", temperature=0.2)
        assert _body(route)["temperature"] == 0.2

    @respx.mock
    def test_sends_max_completion_tokens_when_given(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=BASE_URL, api_key="k", model="m", max_completion_tokens=256)
        assert _body(route)["max_completion_tokens"] == 256

    @respx.mock
    def test_temperature_zero_is_sent_not_dropped(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=BASE_URL, api_key="k", model="m", temperature=0.0)
        assert _body(route)["temperature"] == 0.0

    @respx.mock
    def test_base_url_trailing_slash_is_stripped(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=f"{BASE_URL}/", api_key="k", model="m")
        assert route.calls[0].request.url.path == "/v1/chat/completions"


# ---------------------------------------------------------------------------
# 响应处理
# ---------------------------------------------------------------------------


class TestChatResponse:
    @respx.mock
    def test_returns_message_content(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response("三清山今天晴。")
        )
        result = chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert result == "三清山今天晴。"

    @respx.mock
    def test_error_status_raises_runtime_error(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "invalid api key"}}
            )
        )
        with pytest.raises(RuntimeError) as exc:
            chat("嗨", base_url=BASE_URL, api_key="bad", model="m")
        assert "API 401" in str(exc.value)
        assert "invalid api key" in str(exc.value)

    @respx.mock
    def test_server_error_raises_runtime_error(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(500, text="boom")
        )
        with pytest.raises(RuntimeError, match="API 500"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")
