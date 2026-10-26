"""``agent_study.llm_client`` 的单元测试。

用 ``respx`` 给 ``httpx`` 打桩，只验证请求契约与错误处理，不联网。
"""
from __future__ import annotations

import json
from typing import Literal

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

from agent_study.llm_client import (
    LLMResponseError,
    LLMResponseTruncatedError,
    LLMStructuredOutputError,
    chat,
    chat_response,
    chat_structured,
)
from agent_study.structured_output import TaskSchema

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


# ---------------------------------------------------------------------------
# 响应结构校验
# ---------------------------------------------------------------------------


class TestChatResponseValidation:
    @respx.mock
    def test_chat_response_exposes_finish_reason_and_usage(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        response = chat_response("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert response.content == "三清山今天晴。"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 20
        assert response.truncated is False

    @respx.mock
    def test_body_is_not_a_json_object(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=["not", "a", "dict"])
        )
        with pytest.raises(LLMResponseError, match="不是 JSON 对象"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_missing_choices(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json={"id": "x"})
        )
        with pytest.raises(LLMResponseError, match="缺少 choices"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_empty_choices(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        with pytest.raises(LLMResponseError, match="choices 为空"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_choice_is_not_an_object(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": ["oops"]})
        )
        with pytest.raises(LLMResponseError, match="choices\\[0\\] 不是"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_missing_message(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"index": 0}]})
        )
        with pytest.raises(LLMResponseError, match="缺少 message"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_missing_content(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant"},
                        }
                    ]
                },
            )
        )
        with pytest.raises(LLMResponseError, match="缺少 message.content"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_truncated_without_content_is_reported_as_truncated(self):
        """思考 token 吃满输出预算时，服务端只回 reasoning_content，正文字段整个没有。"""
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "想了一下",
                            },
                        }
                    ]
                },
            )
        )
        with pytest.raises(LLMResponseTruncatedError, match="截断"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")


class TestTruncatedResponse:
    def _truncated(self) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "半截答"},
                    }
                ]
            },
        )

    @respx.mock
    def test_truncated_content_raises_in_chat(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=self._truncated()
        )
        with pytest.raises(LLMResponseTruncatedError, match="截断"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_truncated_error_is_a_response_error(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=self._truncated()
        )
        with pytest.raises(LLMResponseError):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_chat_response_returns_truncated_content_with_flag(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=self._truncated()
        )
        response = chat_response("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert response.content == "半截答"
        assert response.finish_reason == "length"
        assert response.truncated is True

    @respx.mock
    def test_content_filter_raises(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "content_filter",
                            "message": {"role": "assistant", "content": ""},
                        }
                    ]
                },
            )
        )
        with pytest.raises(LLMResponseError, match="内容过滤器"):
            chat("嗨", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_missing_usage_becomes_empty_dict(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "好"},
                        }
                    ]
                },
            )
        )
        response = chat_response("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert response.usage == {}


# ---------------------------------------------------------------------------
# 结构化输出
# ---------------------------------------------------------------------------

TASK_JSON = {
    "title": "写周报",
    "description": "整理本周进展并同步给团队",
    "priority": "high",
    "tags": ["docs"],
    "estimated_minutes": 60,
}


def _structured_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"total_tokens": 30},
        },
    )


class TestChatStructuredRequest:
    @respx.mock
    def test_posts_to_chat_completions_endpoint(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        assert route.calls[0].request.url.path == "/v1/chat/completions"

    @respx.mock
    def test_json_prompt_is_serialized_into_the_message(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured(
            {"summary": "写周报", "minutes": 60},
            base_url=BASE_URL,
            api_key="k",
            model="m",
        )
        content = _body(route)["messages"][0]["content"]
        assert '"summary": "写周报"' in content
        assert '"minutes": 60' in content

    @respx.mock
    def test_message_contains_the_json_schema(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        content = _body(route)["messages"][0]["content"]
        assert "JSON Schema" in content
        assert "estimated_minutes" in content
        assert "priority" in content

    @respx.mock
    def test_json_string_prompt_is_treated_as_json_input(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured(
            json.dumps({"summary": "写周报", "minutes": 60}),
            base_url=BASE_URL,
            api_key="k",
            model="m",
        )
        content = _body(route)["messages"][0]["content"]
        assert '"summary": "写周报"' in content

    @respx.mock
    def test_plain_text_prompt_is_kept_as_is(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        content = _body(route)["messages"][0]["content"]
        assert content.endswith("输入：\n写周报")

    @respx.mock
    def test_default_instruction_tells_model_to_extract(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        content = _body(route)["messages"][0]["content"]
        assert "抽取" in content
        assert "不要原样照抄输入" in content

    @respx.mock
    def test_custom_instruction_is_used(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured(
            "写周报",
            instruction="把输入改写成三个要点，再填进对象",
            base_url=BASE_URL,
            api_key="k",
            model="m",
        )
        content = _body(route)["messages"][0]["content"]
        assert content.startswith("把输入改写成三个要点，再填进对象")
        assert "抽取" not in content

    @respx.mock
    def test_omits_response_format_by_default(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        assert "response_format" not in _body(route)

    @respx.mock
    def test_sends_response_format_when_given(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured(
            "写周报",
            base_url=BASE_URL,
            api_key="k",
            model="m",
            response_format="json_object",
        )
        assert _body(route)["response_format"] == {"type": "json_object"}


class TestChatStructuredResponse:
    @respx.mock
    def test_returns_validated_model_instance(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        task = chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        assert isinstance(task, TaskSchema)
        assert task.title == "写周报"
        assert task.priority == "high"
        assert task.estimated_minutes == 60
        assert task.tags == ["docs"]

    @respx.mock
    def test_accepts_json_wrapped_in_markdown_fence(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(
                "```json\n" + json.dumps(TASK_JSON) + "\n```"
            )
        )
        task = chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        assert task.title == "写周报"

    @respx.mock
    def test_uses_custom_schema(self):
        class Verdict(BaseModel):
            label: Literal["pos", "neg"]
            score: float = Field(ge=0.0, le=1.0)

        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(
                json.dumps({"label": "pos", "score": 0.9})
            )
        )
        verdict = chat_structured(
            "判断情感", schema=Verdict, base_url=BASE_URL, api_key="k", model="m"
        )
        assert isinstance(verdict, Verdict)
        assert verdict.label == "pos"
        assert verdict.score == 0.9

    @respx.mock
    def test_invalid_json_raises(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response("这不是 JSON")
        )
        with pytest.raises(LLMStructuredOutputError, match="不是合法 JSON"):
            chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_json_array_raises(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps([TASK_JSON]))
        )
        with pytest.raises(LLMStructuredOutputError, match="不是 JSON 对象"):
            chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_schema_violation_raises(self):
        bad = dict(TASK_JSON, priority="urgent")
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(bad))
        )
        with pytest.raises(LLMStructuredOutputError, match="不符合 TaskSchema"):
            chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_truncated_response_raises_truncated_error(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length",
                            "message": {"role": "assistant", "content": '{"title"'},
                        }
                    ]
                },
            )
        )
        with pytest.raises(LLMResponseTruncatedError, match="截断"):
            chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")

    @respx.mock
    def test_error_status_raises_runtime_error(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
        )
        with pytest.raises(RuntimeError, match="API 401"):
            chat_structured("写周报", base_url=BASE_URL, api_key="bad", model="m")
