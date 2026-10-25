"""MiniMax-M3.1-Flash 的 Anthropic SDK 调用示例（对照 ``llm_client.py`` 的 OpenAI 兼容写法）。

同一个模型、同一把 Key，换一种协议形态。两边的概念映射：

======================  ==============================  ==================================
概念                    OpenAI 兼容（``llm_client.py``）   Anthropic Messages（本模块）
======================  ==============================  ==================================
端点                    ``POST /v1/chat/completions``    ``POST /v1/messages``
system 提示词           ``messages[0].role=system``      顶层 ``system`` 参数
输出上限                ``max_completion_tokens``        ``max_tokens``（**必填**）
推理强度                ``reasoning_effort``             ``output_config.effort``
结束原因                ``finish_reason``                ``stop_reason``
思考内容                ``reasoning_content`` 字段        ``thinking`` 类型的 content block
正文位置                ``choices[0].message.content``   ``content`` 里的 ``text`` block
======================  ==============================  ==================================

用法::

    from agent_study.anthropic_chat import chat

    print(chat("用一句话解释 Agent"))

运行前在 ``.env.dev`` 里配置好 ``API_KEY``（MiniMax 订阅 Key）；
``ANTHROPIC_BASE_URL`` 留空则回退到 ``API_BASE_URL``。
"""

from __future__ import annotations

import anthropic

from agent_study.config import settings

#: Anthropic Messages API 要求显式给出输出上限，不给会直接 400
DEFAULT_MAX_TOKENS = 1024


def chat_message(
    prompt: str,
    *,
    system: str | None = None,
    base_url: str = "",
    api_key: str = "",
    model: str = settings.MODEL_NAME,
    reasoning_effort: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 120.0,
) -> anthropic.types.Message:
    """调用 Anthropic Messages API，返回完整响应（含 ``stop_reason`` 和 ``usage``）。

    ``stop_reason`` 必须检查：值为 ``max_tokens`` 表示输出被截断，
    此时正文是半截的，但 HTTP 状态码仍然是 200。
    """
    client = anthropic.Anthropic(
        api_key=api_key or settings.API_KEY,
        base_url=base_url or settings.ANTHROPIC_BASE_URL or settings.API_BASE_URL,
        timeout=timeout,
    )

    kwargs: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if reasoning_effort:
        kwargs["output_config"] = {"effort": reasoning_effort}

    try:
        return client.messages.create(**kwargs)  # type: ignore[arg-type]
    except anthropic.APIStatusError as exc:
        raise RuntimeError(
            f"API {exc.status_code}: {exc.response.text[:300]}"
        ) from exc


def chat(
    prompt: str,
    *,
    system: str | None = None,
    base_url: str = "",
    api_key: str = "",
    model: str = settings.MODEL_NAME,
    reasoning_effort: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 120.0,
) -> str:
    """调用 Anthropic Messages API，返回正文文本。

    与 ``llm_client.chat`` 的区别：Anthropic 协议里 ``max_tokens`` 是必填项，
    且正文要自己从 ``content`` 的 text block 里拼出来。
    需要 ``stop_reason`` / ``usage`` 时改用 :func:`chat_message`。
    """
    message = chat_message(
        prompt,
        system=system,
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return "".join(block.text for block in message.content if block.type == "text")
