"""MiniMax-M3.1-Flash 最小调用示例（OpenAI 兼容接口）。

官方参数（见 MiniMax 开放平台文档）：
- base_url: ``https://api.minimax.io/v1``
- model:    ``MiniMax-M3.1-Flash``（1M 上下文）
- 思考深度 ``reasoning_effort``: low / medium / high / xhigh / max（不传由服务端取默认）
- OpenAI 兼容协议下，思考内容在 ``reasoning_content`` 字段，``content`` 只有最终答案

用法::

    from agent_study.llm_client import chat, chat_response

    print(chat("用一句话解释 Agent"))

    response = chat_response("用一句话解释 Agent")
    print(response.content, response.finish_reason, response.usage)

结构化输出走 :func:`chat_structured`，prompt 可以直接给 JSON 数据：

    from agent_study.llm_client import chat_structured

    task = chat_structured({"summary": "写周报", "minutes": 60})
    print(task.title, task.priority, task.estimated_minutes)

需要同时拿到响应元信息（``finish_reason`` / ``usage``）时用
:func:`chat_structured_response`：

    from agent_study.llm_client import chat_structured_response

    task, response = chat_structured_response({"summary": "写周报"})
    print(task.title, response.finish_reason, response.usage)

运行前在 ``.env.dev`` 里配置好 ``API_KEY``（MiniMax 订阅 Key）；
``API_BASE_URL`` 留空则使用官方地址。

``temperature`` 与 ``max_completion_tokens`` 默认都不发送，
换用支持它们的模型（如 OpenAI 官方模型）时再显式传。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agent_study.config import settings
from agent_study.structured_output import TaskSchema

ModelT = TypeVar("ModelT", bound=BaseModel)

DEFAULT_STRUCTURED_INSTRUCTION = (
    "请根据下面的输入内容抽取并填写一个对象，"
    "信息缺失时按输入合理推断，不要原样照抄输入。"
)


class LLMResponseError(RuntimeError):
    """LLM 响应结构不符合预期（HTTP 成功，但内容没法用）。"""


class LLMResponseTruncatedError(LLMResponseError):
    """输出被 ``max_completion_tokens`` 截断，正文只有半截。"""


class LLMStructuredOutputError(LLMResponseError):
    """模型输出不是合法 JSON，或不符合约定的 JSON Schema。"""


@dataclass(frozen=True)
class ChatResponse:
    """一次 chat completion 响应里值得拿走的字段。

    ``finish_reason`` 和 ``usage`` 是一等返回值：前者说明"为什么停下来"，
    后者是计费和成本估算的唯一权威来源。
    """

    content: str
    finish_reason: str
    usage: dict[str, Any]

    @property
    def truncated(self) -> bool:
        """正文是否因为撞到输出上限而被截断。"""
        return self.finish_reason == "length"


_shared_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """取共享的 HTTP 客户端，进程内只建一次。

    每个请求新建 client 会丢掉连接池，每次都要重新握手；
    共享一个 client 才能复用 keep-alive 连接。``httpx.Client`` 本身是
    线程安全的，可以直接跨线程共用。
    """
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.Client()
    return _shared_client


def close_client() -> None:
    """关闭共享的 HTTP 客户端。

    不调用也能正常工作（进程退出时连接会被回收），
    提前关闭可以及时释放套接字。
    """
    global _shared_client
    if _shared_client is not None:
        _shared_client.close()
        _shared_client = None


def chat_response(
    prompt: str,
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = settings.MODEL_NAME,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    timeout: float = 120.0,
) -> ChatResponse:
    """调用 OpenAI 兼容的 chat completions 接口，返回校验后的响应。

    响应结构不符合预期时抛 :class:`LLMResponseError`。
    """
    url = (
        f"{(base_url or settings.API_BASE_URL).rstrip('/')}"
        "/chat/completions"
    )
    payload = _build_payload(
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )

    return _parse_response(_post(url, payload, api_key=api_key, timeout=timeout))


def chat(
    prompt: str,
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = settings.MODEL_NAME,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    timeout: float = 120.0,
) -> str:
    """调用 OpenAI 兼容的 chat completions 接口，返回正文文本。

    ``temperature`` 与 ``max_completion_tokens`` 为 ``None`` 时不会出现在请求体里，
    避免把模型不支持的参数发到线上。

    响应被截断（``finish_reason == "length"``）时抛
    :class:`LLMResponseTruncatedError`，而不是把半截答案当成完整答案返回。
    """
    response = chat_response(
        prompt,
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
    )
    if response.truncated:
        raise LLMResponseTruncatedError(
            "输出被 max_completion_tokens 截断，只有半截正文："
            f"{response.content[:100]!r}"
        )
    if response.finish_reason == "content_filter":
        raise LLMResponseError(
            "内容被内容过滤器拦截（finish_reason=content_filter）"
        )
    return response.content


def chat_structured_response(
    prompt: str | dict[str, Any] | list[Any],
    *,
    schema: type[ModelT] = TaskSchema,  # type: ignore[assignment]
    instruction: str = "",
    base_url: str = "",
    api_key: str = "",
    model: str = settings.MODEL_NAME,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    response_format: str | None = None,
    timeout: float = 120.0,
) -> tuple[ModelT, ChatResponse]:
    """让模型按 JSON Schema 输出，返回校验后的模型实例和原始响应元信息。

    返回 ``(instance, response)``：``instance`` 是按 ``schema`` 校验过的
    Pydantic 对象，``response`` 是 :class:`ChatResponse`，带有
    ``finish_reason`` 和 ``usage``。

    ``prompt`` 既可以是字符串，也可以是 JSON 可序列化的 dict / list（任务数据），
    后者会被序列化后连同 JSON Schema 一起发给模型。

    默认按 :class:`~agent_study.structured_output.TaskSchema` 校验；传别的
    Pydantic 模型即可复用同一个入口。

    ``instruction`` 用来告诉模型"拿输入做什么"。不写时用默认的抽取指令。
    注意：如果输入本身已经是 schema 的合法实例（字段齐全），模型很可能
    原样返回——那不是解析出错，而是它没有可做的变换。遇到这种情况，
    把输入换成原始素材（自由文本、键值不全的 dict），或用 ``instruction``
    明确要求改写。

    模型输出不是合法 JSON、或不符合 schema 时抛
    :class:`LLMStructuredOutputError`；输出被截断时抛
    :class:`LLMResponseTruncatedError`。
    """
    url = (
        f"{(base_url or settings.API_BASE_URL).rstrip('/')}"
        "/chat/completions"
    )
    payload = _build_payload(
        _build_structured_content(prompt, schema, instruction),
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        response_format=response_format,
    )

    response = _parse_response(_post(url, payload, api_key=api_key, timeout=timeout))
    if response.truncated:
        raise LLMResponseTruncatedError(
            "输出被 max_completion_tokens 截断，只有半截正文："
            f"{response.content[:100]!r}"
        )
    if response.finish_reason == "content_filter":
        raise LLMResponseError(
            "内容被内容过滤器拦截（finish_reason=content_filter）"
        )

    data = _extract_json_object(response.content)
    try:
        instance = schema.model_validate(data)
    except ValidationError as e:
        raise LLMStructuredOutputError(
            f"模型输出不符合 {schema.__name__}：{e.error_count()} 处错误，"
            f"原文：{response.content[:200]!r}"
        ) from e

    return instance, response


def chat_structured(
    prompt: str | dict[str, Any] | list[Any],
    *,
    schema: type[ModelT] = TaskSchema,  # type: ignore[assignment]
    instruction: str = "",
    base_url: str = "",
    api_key: str = "",
    model: str = settings.MODEL_NAME,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    response_format: str | None = None,
    timeout: float = 120.0,
) -> ModelT:
    """让模型按 JSON Schema 输出，并返回校验后的模型实例。

    :func:`chat_structured_response` 的便捷包装，只返回模型实例；
    需要 ``finish_reason`` / ``usage`` 时改用那个函数。
    """
    instance, _ = chat_structured_response(
        prompt,
        schema=schema,
        instruction=instruction,
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        response_format=response_format,
        timeout=timeout,
    )
    return instance


def _build_payload(
    content: str,
    *,
    model: str,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    response_format: str | None = None,
) -> dict[str, object]:
    """组装请求体。

    ``temperature`` 与 ``max_completion_tokens`` 为 ``None`` 时不出现在请求体里，
    避免把模型不支持的参数发到线上。
    """
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        payload["temperature"] = temperature
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    if response_format is not None:
        payload["response_format"] = {"type": response_format}
    return payload


def _post(
    url: str,
    payload: dict[str, object],
    *,
    api_key: str,
    timeout: float,
) -> object:
    """发送请求并返回已解析的 JSON body。"""
    headers = {"Authorization": f"Bearer {api_key or settings.API_KEY}"}
    response = _get_client().post(
        url, json=payload, headers=headers, timeout=timeout
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"API {response.status_code}: {response.text[:300]}"
        )
    return response.json()


def _build_structured_content(
    prompt: str | dict[str, Any] | list[Any],
    schema: type[BaseModel],
    instruction: str = "",
) -> str:
    """把用户输入和 JSON Schema 拼成一条请求消息。

    dict / list 输入按 JSON 序列化；字符串如果本身是合法 JSON
    （``model_dump_json()`` 的输出），也按 JSON 结构化后再发送，
    让模型看到和 schema 对齐的字段，而不是一坨裸文本。

    ``instruction`` 放在最前面，先说清"做什么"再说"按什么格式"，
    否则模型只看到一个格式约束，容易把输入原样抄回来。
    """
    user_input = _render_input(prompt)
    schema_json = json.dumps(
        schema.model_json_schema(), ensure_ascii=False, indent=2
    )
    task = instruction.strip() or DEFAULT_STRUCTURED_INSTRUCTION
    return (
        f"{task}\n\n"
        "请只输出一个 JSON 对象，严格符合下面的 JSON Schema，"
        "不要输出解释文字，也不要用 markdown 代码块包裹。\n"
        f"{schema_json}\n\n"
        "输入：\n"
        f"{user_input}"
    )


def _render_input(prompt: str | dict[str, Any] | list[Any]) -> str:
    """把 prompt 渲染成发给模型的文本。"""
    if not isinstance(prompt, str):
        return json.dumps(prompt, ensure_ascii=False, indent=2)

    text = prompt.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return prompt
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False, indent=2)
    return prompt


def _extract_json_object(content: str) -> dict[str, Any]:
    """从模型输出里取出 JSON 对象。

    容忍 ```json 代码块包裹（模型很爱加），但不接受 JSON 数组或裸文本。
    """
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMStructuredOutputError(
            f"模型输出不是合法 JSON：{content[:200]!r}"
        ) from e

    if not isinstance(data, dict):
        raise LLMStructuredOutputError(
            f"模型输出不是 JSON 对象：{type(data).__name__}"
        )
    return data


def _parse_response(data: object) -> ChatResponse:
    """校验响应结构，取出正文、结束原因和用量。

    这里不做业务判断（截断、内容过滤），只保证"拿到的就是想要的那几个字段"。
    """
    if not isinstance(data, dict):
        raise LLMResponseError(
            f"响应不是 JSON 对象：{type(data).__name__}"
        )

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("响应缺少 choices，或 choices 为空")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMResponseError("choices[0] 不是 JSON 对象")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise LLMResponseError("choices[0] 缺少 message 对象")

    content = message.get("content")
    finish_reason = choice.get("finish_reason")
    if content is None:
        # 思考 token 吃满输出预算时，服务端只回 reasoning_content，正文整个字段都没有
        if finish_reason == "length":
            raise LLMResponseTruncatedError(
                "输出被 max_completion_tokens 截断，服务端没有返回正文"
            )
        raise LLMResponseError("响应缺少 message.content")

    usage = data.get("usage")

    return ChatResponse(
        content=str(content),
        finish_reason=str(finish_reason) if finish_reason is not None else "",
        usage=usage if isinstance(usage, dict) else {},
    )
