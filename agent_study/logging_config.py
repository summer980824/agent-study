import contextvars
import json
import logging

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
            "username",
            "duration_ms",
            "status_code",
            "error_type",
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
    """初始化 agent_study 的日志输出。"""

    logger = logging.getLogger("agent_study")

    if not any(
        getattr(handler, "_agent_study_handler", False)
        for handler in logger.handlers
    ):
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