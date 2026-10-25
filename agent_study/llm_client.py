"""MiniMax-M3.1-Flash 最小调用示例（OpenAI 兼容接口）。

官方参数（见 MiniMax 开放平台文档）：
- base_url: ``https://api.minimax.io/v1``
- model:    ``MiniMax-M3.1-Flash``（1M 上下文）
- 思考深度 ``reasoning_effort``: low / medium / high / xhigh / max（不传由服务端取默认）
- OpenAI 兼容协议下，思考内容在 ``reasoning_content`` 字段，``content`` 只有最终答案

用法::

    from agent_study.llm_client import chat

    print(chat("用一句话解释 Agent"))

运行前在 ``.env.dev`` 里配置好 ``API_KEY``（MiniMax 订阅 Key）；
``API_BASE_URL`` 留空则使用官方地址。
"""

from __future__ import annotations

import httpx

from agent_study.config import settings


def chat(
    prompt: str,
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = settings.MODEL_NAME,
    reasoning_effort: str | None = None,
    timeout: float = 120.0,
) -> str:
    url = (
        f"{(base_url or settings.API_BASE_URL).rstrip('/')}"
        "/chat/completions"
    )
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    headers = {"Authorization": f"Bearer {api_key or settings.API_KEY}"}
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise RuntimeError(
                f"API {response.status_code}: {response.text[:300]}"
            )
        data = response.json()

    return str(data["choices"][0]["message"]["content"])

