"""``agent_study.anthropic_chat`` 的单元测试。

Anthropic SDK 2.x 底层用的是 ``httpx2``，而项目里的 ``respx`` 只支持 ``httpx`` v1，
拦不到 SDK 的请求，所以这里改用 ``httpx2`` 自带的 ``MockTransport`` 打桩，
并把它注入 SDK 的 ``http_client``。全程不联网。
"""
from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest

from agent_study import anthropic_chat
from agent_study.anthropic_chat import chat, chat_message

BASE_URL = "https://api.example.com/anthropic"

MESSAGE_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "test-model",
    "content": [
        {"type": "thinking", "thinking": "想一下", "signature": "sig"},
        {"type": "text", "text": "三清山今天晴。"},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 8},
}


class _MockAnthropicAPI:
    """记录 SDK 发出的请求，并按预设响应回包。"""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}
        self.status_code = 200
        self.response_json: dict[str, Any] = MESSAGE_RESPONSE

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.captured["request"] = request
        self.captured["body"] = json.loads(request.content)
        return httpx2.Response(self.status_code, json=self.response_json)

    @property
    def body(self) -> dict:
        return self.captured["body"]


@pytest.fixture
def mock_api(monkeypatch: pytest.MonkeyPatch) -> _MockAnthropicAPI:
    api = _MockAnthropicAPI()
    transport = httpx2.MockTransport(api.handler)
    real_cls = anthropic.Anthropic

    def factory(**kwargs: Any) -> anthropic.Anthropic:
        kwargs["http_client"] = httpx2.Client(transport=transport)
        return real_cls(**kwargs)

    monkeypatch.setattr(anthropic_chat.anthropic, "Anthropic", factory)
    return api


# ---------------------------------------------------------------------------
# 请求契约
# ---------------------------------------------------------------------------


class TestChatRequest:
    def test_posts_to_messages_endpoint(self, mock_api):
        chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert mock_api.captured["request"].url.path == "/anthropic/v1/messages"

    def test_sends_bearer_authorization_header(self, mock_api):
        chat("嗨", base_url=BASE_URL, api_key="sk-secret", model="m")
        headers = mock_api.captured["request"].headers
        assert headers["authorization"] == "Bearer sk-secret"

    def test_sends_model_and_messages(self, mock_api):
        chat("今天三清山天气如何？", base_url=BASE_URL, api_key="k", model="m")
        assert mock_api.body["model"] == "m"
        assert mock_api.body["messages"] == [
            {"role": "user", "content": "今天三清山天气如何？"}
        ]

    def test_max_tokens_is_always_present(self, mock_api):
        chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert mock_api.body["max_tokens"] == anthropic_chat.DEFAULT_MAX_TOKENS

    def test_omits_optional_params_by_default(self, mock_api):
        chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        body = mock_api.body
        assert "system" not in body
        assert "output_config" not in body
        assert "temperature" not in body

    def test_sends_system_as_top_level_param(self, mock_api):
        chat("嗨", base_url=BASE_URL, api_key="k", model="m", system="你是天气助手")
        assert mock_api.body["system"] == "你是天气助手"

    def test_sends_reasoning_effort_as_output_config(self, mock_api):
        chat("嗨", base_url=BASE_URL, api_key="k", model="m", reasoning_effort="low")
        assert mock_api.body["output_config"] == {"effort": "low"}

    def test_sends_temperature_when_given(self, mock_api):
        chat("嗨", base_url=BASE_URL, api_key="k", model="m", temperature=0.2)
        assert mock_api.body["temperature"] == 0.2

    def test_temperature_zero_is_sent_not_dropped(self, mock_api):
        chat("嗨", base_url=BASE_URL, api_key="k", model="m", temperature=0.0)
        assert mock_api.body["temperature"] == 0.0

    def test_max_completion_tokens_maps_to_max_tokens(self, mock_api):
        chat("嗨", base_url=BASE_URL, api_key="k", model="m", max_completion_tokens=256)
        assert mock_api.body["max_tokens"] == 256

    def test_base_url_trailing_slash_is_stripped(self, mock_api):
        chat("嗨", base_url=f"{BASE_URL}/", api_key="k", model="m")
        assert mock_api.captured["request"].url.path == "/anthropic/v1/messages"


# ---------------------------------------------------------------------------
# 响应处理
# ---------------------------------------------------------------------------


class TestChatResponse:
    def test_returns_only_text_blocks(self, mock_api):
        result = chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert result == "三清山今天晴。"

    def test_chat_message_exposes_stop_reason_and_usage(self, mock_api):
        message = chat_message("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert message.stop_reason == "end_turn"
        assert message.usage.input_tokens == 12
        assert message.usage.output_tokens == 8

    def test_error_status_raises_runtime_error(self, mock_api):
        mock_api.status_code = 401
        mock_api.response_json = {"error": {"message": "invalid api key"}}
        with pytest.raises(RuntimeError) as exc:
            chat("嗨", base_url=BASE_URL, api_key="bad", model="m")
        assert "API 401" in str(exc.value)
        assert "invalid api key" in str(exc.value)

    def test_server_error_raises_runtime_error(self, mock_api):
        mock_api.status_code = 500
        mock_api.response_json = {"error": {"message": "boom"}}
        with pytest.raises(RuntimeError, match="API 500"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")
