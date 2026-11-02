import contextvars
import json
import logging
import sys

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


_request_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("request_id", default=None)
)


class JsonFormatter(logging.Formatter):
    """将日志记录转换为 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": (
                getattr(record, "request_id", None)
                or _request_id.get()
            ),
        }

        # 仅输出明确允许的业务上下文字段
        for key in (
            "event",
            "model",
            "username",
            "duration_ms",
            "status_code",
            "error_type",
            "finish_reason",
            "usage",
        ):
            value = getattr(record, key, None)
            if value is not None:
                data[key] = value

        if record.exc_info:
            data["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(data, ensure_ascii=False)


def configure_logging(
    level: int = logging.INFO,
) -> None:
    """初始化 agent_study 的日志输出。

    绑定到当前的 ``sys.stderr``：宿主（测试框架、进程重定向）换掉
    stderr 后，绑在旧流上的 handler 只会写出 ``--- Logging error ---``。
    所以 handler 只复用"还挂在当前 stderr 上"的那个，否则重建。
    """

    logger = logging.getLogger("agent_study")

    handler = next(
        (h for h in logger.handlers if getattr(h, "_agent_study_handler", False)),
        None,
    )
    if handler is None or handler.stream is not sys.stderr:
        if handler is not None:
            logger.removeHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._agent_study_handler = True
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False


@contextmanager
def request_context(
    request_id: str,
) -> Iterator[None]:
    """在当前执行上下文中设置请求 ID。"""

    token = _request_id.set(request_id)

    try:
        yield
    finally:
        _request_id.reset(token)
