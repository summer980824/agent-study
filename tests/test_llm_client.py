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

from agent_study import llm_client
from agent_study.llm_client import (
    ChatResponse,
    ChatResult,
    ChatStream,
    LLMResponseError,
    LLMResponseTruncatedError,
    LLMStructuredOutputError,
    _get_client,
    chat,
    chat_response,
    chat_stream,
    chat_structured,
    chat_structured_response,
    close_client,
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
    def test_message_forbids_additional_properties(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        content = _body(route)["messages"][0]["content"]
        assert '"additionalProperties": false' in content

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


class TestChatStructuredResult:
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
    def test_returns_only_the_model_without_response_metadata(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        task = chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        assert isinstance(task, TaskSchema)
        assert task.title == "写周报"

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
    def test_extra_field_raises(self):
        bad = dict(TASK_JSON, due_date="2026-10-26")
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


class TestChatStructuredResponse:
    @respx.mock
    def test_returns_model_instance_and_chat_response(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        task, response = chat_structured_response(
            "写周报", base_url=BASE_URL, api_key="k", model="m"
        )
        assert isinstance(task, TaskSchema)
        assert task.title == "写周报"
        assert isinstance(response, ChatResponse)

    @respx.mock
    def test_exposes_finish_reason_and_usage(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        _, response = chat_structured_response(
            "写周报", base_url=BASE_URL, api_key="k", model="m"
        )
        assert response.finish_reason == "stop"
        assert response.usage == {"total_tokens": 30}
        assert response.truncated is False

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
            chat_structured_response(
                "写周报", base_url=BASE_URL, api_key="k", model="m"
            )

    @respx.mock
    def test_chat_structured_returns_the_same_instance(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_structured_response(json.dumps(TASK_JSON))
        )
        instance, _ = chat_structured_response(
            "写周报", base_url=BASE_URL, api_key="k", model="m"
        )
        task = chat_structured("写周报", base_url=BASE_URL, api_key="k", model="m")
        assert task == instance


# ---------------------------------------------------------------------------
# 共享 HTTP 客户端
# ---------------------------------------------------------------------------


class TestSharedHttpClient:
    def test_client_instance_is_reused(self):
        assert _get_client() is _get_client()

    def test_close_client_closes_and_allows_a_new_client(self):
        first = _get_client()
        close_client()
        assert first.is_closed
        second = _get_client()
        assert second is not first
        close_client()

    @respx.mock
    def test_requests_do_not_create_a_client_each_time(self, monkeypatch):
        created: list[httpx.Client] = []
        original_init = httpx.Client.__init__

        def counting_init(self, *args, **kwargs):
            created.append(self)
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", counting_init)
        close_client()
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        close_client()
        assert len(created) == 1


# ---------------------------------------------------------------------------
# 流式输出
# ---------------------------------------------------------------------------

SSE_BODY = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":9}}\n\n'
    "data: [DONE]\n\n"
)


def _sse_response(body: str = SSE_BODY, status: int = 200) -> httpx.Response:
    """构造一个 SSE 文本流响应。

    SSE body 必须自带 ``\\n\\n`` 分隔符和结尾换行，``httpx.Response(text=...)``
    不会替我们补上。
    """
    return httpx.Response(
        status,
        text=body,
        headers={"Content-Type": "text/event-stream"},
    )


class _RawStream(httpx.SyncByteStream):
    """未经 httpx 预读的字节流，用来复现真实网络响应下 body 未读的状态。"""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self):
        yield self._data


class TestChatStreamRequest:
    @respx.mock
    def test_enables_streaming_on_the_request_body(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        list(chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m"))
        assert _body(route)["stream"] is True

    @respx.mock
    def test_sends_bearer_authorization_header(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        list(chat_stream("嗨", base_url=BASE_URL, api_key="sk-secret", model="m"))
        assert route.calls[0].request.headers["authorization"] == "Bearer sk-secret"

    @respx.mock
    def test_asks_for_an_event_stream(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        list(chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m"))
        assert route.calls[0].request.headers["accept"] == "text/event-stream"

    @respx.mock
    def test_omits_stream_options_by_default(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        list(chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m"))
        assert "stream_options" not in _body(route)

    @respx.mock
    def test_sends_stream_options_when_include_usage(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        list(
            chat_stream(
                "嗨",
                base_url=BASE_URL,
                api_key="k",
                model="m",
                include_usage=True,
            )
        )
        assert _body(route)["stream_options"] == {"include_usage": True}

    @respx.mock
    def test_sync_requests_are_not_switched_to_streaming(self):
        """``stream`` 只加在流式请求上，不能污染同步请求体。"""
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response()
        )
        chat("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert "stream" not in _body(route)


class TestChatStreamResult:
    @respx.mock
    def test_yields_content_deltas_in_order(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["你", "好"]

    @respx.mock
    def test_result_accumulates_content(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        list(stream)
        assert stream.result.content == "你好"

    @respx.mock
    def test_result_carries_finish_reason_and_usage(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        list(stream)
        assert stream.result.finish_reason == "stop"
        assert stream.result.usage == {"total_tokens": 9}
        assert stream.result.truncated is False

    @respx.mock
    def test_result_is_the_same_object_as_the_chat_result_type(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert isinstance(stream.result, ChatResult)
        assert ChatResponse is ChatResult

    @respx.mock
    def test_content_accumulates_while_iterating(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        seen: list[str] = []
        for delta in stream:
            seen.append(delta)
            assert stream.result.content == "".join(seen)

    @respx.mock
    def test_reasoning_content_is_accumulated_separately(self):
        body = (
            'data: {"choices":[{"index":0,'
            '"delta":{"reasoning_content":"想一下"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"答"},"finish_reason":null}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["答"]
        assert stream.result.reasoning_content == "想一下"
        assert stream.result.content == "答"

    @respx.mock
    def test_missing_usage_becomes_empty_dict(self):
        body = (
            'data: {"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        list(stream)
        assert stream.result.usage == {}


class TestChatStreamParsing:
    @respx.mock
    def test_skips_blank_and_non_data_lines(self):
        body = (
            "\n"
            ": keep-alive comment\n"
            "event: message\n"
            'data: {"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":"stop"}]}\n'
            "\n"
            "data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["好"]

    @respx.mock
    def test_handles_data_prefix_without_a_space(self):
        body = (
            'data:{"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":"stop"}]}\n'
            "\n"
            "data:[DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["好"]
        assert stream.result.finish_reason == "stop"

    @respx.mock
    def test_skips_unparsable_lines_and_keeps_going(self):
        body = (
            "data: {not json}\n\n"
            'data: {"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":"stop"}]}\n'
            "\n"
            "data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["好"]

    @respx.mock
    def test_usage_only_chunk_without_choices(self):
        """OpenAI 兼容协议里带 usage 的尾块常常没有 choices，不能因此丢掉 usage。"""
        body = (
            'data: {"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":"stop"}]}\n'
            "\n"
            'data: {"choices":[],"usage":{"prompt_tokens":3,"total_tokens":11}}\n'
            "\n"
            "data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["好"]
        assert stream.result.usage == {"prompt_tokens": 3, "total_tokens": 11}

    @respx.mock
    def test_chunk_without_delta_is_skipped(self):
        body = (
            'data: {"choices":[{"index":0,"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":null,"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":"stop"}]}\n'
            "\n"
            "data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["好"]

    @respx.mock
    def test_in_stream_error_payload_raises(self):
        body = 'data: {"error":{"message":"upstream boom"}}\n\n'
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        with pytest.raises(RuntimeError, match="upstream boom"):
            list(stream)

    @respx.mock
    def test_done_terminates_the_stream(self):
        body = (
            'data: {"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":"stop"}]}\n'
            "\n"
            "data: [DONE]\n\n"
            'data: {"choices":[{"index":0,"delta":{"content":"不该出现"},"finish_reason":null}]}\n'
            "\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["好"]

    @respx.mock
    def test_stream_without_done_marker_keeps_partial_content(self):
        body = (
            'data: {"choices":[{"index":0,"delta":{"content":"半"},"finish_reason":null}]}\n'
            "\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["半"]
        assert stream.result.content == "半"
        assert stream.result.finish_reason == ""


class TestChatStreamErrors:
    @respx.mock
    def test_error_status_raises_runtime_error_at_call_time(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "invalid api key"}})
        )
        with pytest.raises(RuntimeError) as exc:
            chat_stream("嗨", base_url=BASE_URL, api_key="bad", model="m")
        assert "API 401" in str(exc.value)
        assert "invalid api key" in str(exc.value)

    def test_error_body_is_read_from_an_unread_streaming_response(self, monkeypatch):
        """真实网络响应下 body 没被预读，取错误信息必须先 read()。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401, stream=_RawStream(b'{"error": "invalid api key"}')
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(llm_client, "_get_client", lambda: client)

        with pytest.raises(RuntimeError) as exc:
            chat_stream("嗨", base_url=BASE_URL, api_key="bad", model="m")
        assert "API 401" in str(exc.value)
        assert "invalid api key" in str(exc.value)

    @respx.mock
    def test_truncated_stream_does_not_raise(self):
        body = (
            'data: {"choices":[{"index":0,"delta":{"content":"半截"},"finish_reason":"length"}]}\n'
            "\n"
            "data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response(body)
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert list(stream) == ["半截"]
        assert stream.result.truncated is True
        assert stream.result.finish_reason == "length"


class TestChatStreamInterface:
    @respx.mock
    def test_iter_returns_self_so_a_second_loop_does_not_resend(self):
        route = respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert iter(stream) is stream
        list(stream)
        list(stream)
        assert len(route.calls) == 1

    @respx.mock
    def test_close_releases_the_connection(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        for _ in stream:
            break
        stream.close()
        stream.close()

    @respx.mock
    def test_is_an_iterator_of_strings(self):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_sse_response()
        )
        stream = chat_stream("嗨", base_url=BASE_URL, api_key="k", model="m")
        assert isinstance(stream, ChatStream)
        assert next(iter(stream)) == "你"

