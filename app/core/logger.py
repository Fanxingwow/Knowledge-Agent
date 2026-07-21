from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger as _logger

from app.core.request_context import get_request_id
from app.core.settings import LoggingSettings

if TYPE_CHECKING:
    from loguru import Record

logger = _logger


def _patch_request_id(record: Record) -> None:
    extra = record["extra"]
    bound_request_id = extra.get("request_id")
    extra["request_id"] = get_request_id() or (
        bound_request_id if isinstance(bound_request_id, str) and bound_request_id else "-"
    )


def configure_logging(settings: LoggingSettings) -> None:
    logger.remove()
    logger.configure(extra={"request_id": "-"}, patcher=_patch_request_id)
    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "request_id={extra[request_id]} | {name}:{function}:{line} | {message}"
    )
    common: dict[str, Any] = {
        "format": log_format,
        "backtrace": False,
        "diagnose": False,
        "serialize": settings.json_output,
    }
    if settings.console_enabled:
        logger.add(sys.stderr, level=settings.console_level, **common)
    if settings.file_enabled:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "application_{time:YYYY-MM-DD}.log",
            level=settings.file_level,
            rotation="00:00",
            retention=settings.retention,
            encoding="utf-8",
            **common,
        )
