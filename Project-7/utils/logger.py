"""
Centralised structured logging configuration using Loguru.

Importing this module configures Loguru with:
    * A colourised console sink for local development.
    * A rotating file sink (`logs/travel_assistant.log`) for persistent,
      structured logs suitable for observability tooling.
    * A dedicated helper, `get_logger`, that returns a logger bound with a
      `component` field so every log line can be traced back to the
      subsystem that emitted it (agent, tool, service, database, ...).

The module also exposes `log_tool_execution`, a decorator used by every
LangChain tool to provide consistent execution tracing: start/end
timestamps, arguments, results, and error tracking.
"""

from __future__ import annotations

import functools
import sys
import time
from typing import Any, Callable, TypeVar

from loguru import logger as _logger

from configs import settings

# ---------------------------------------------------------------------------
# Base configuration
# ---------------------------------------------------------------------------
_logger.remove()  # Remove the default handler so we control formatting.

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[component]}</cyan> | "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{extra[component]} | {name}:{function}:{line} | {message}"
)

_logger.configure(extra={"component": "app"})

_logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format=_CONSOLE_FORMAT,
    colorize=True,
    backtrace=False,
    diagnose=False,
)

_logger.add(
    str(settings.LOG_FILE),
    level=settings.LOG_LEVEL,
    format=_FILE_FORMAT,
    rotation=settings.LOG_ROTATION,
    retention=settings.LOG_RETENTION,
    enqueue=True,
    backtrace=True,
    diagnose=False,
)


def get_logger(component: str):
    """Return a Loguru logger bound to a specific component name.

    Args:
        component: A short identifier of the subsystem (e.g. ``"flight_tool"``,
            ``"travel_agent"``, ``"database"``).

    Returns:
        A Loguru logger instance with the ``component`` field bound, used to
        tag every emitted record.
    """
    return _logger.bind(component=component)


F = TypeVar("F", bound=Callable[..., Any])


def log_tool_execution(component: str) -> Callable[[F], F]:
    """Decorator that adds structured execution tracing to LangChain tools.

    Logs the tool name, input arguments, execution duration, a truncated
    preview of the result, and full exception details on failure. This
    provides the "Tool Usage Logs" and "Execution Tracing" required for
    observability.

    Args:
        component: Name of the component/tool being traced.
    """

    def decorator(func: F) -> F:
        bound_logger = get_logger(component)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            bound_logger.info(
                "Tool invoked | args={} kwargs={}",
                _safe_repr(args),
                _safe_repr(kwargs),
            )
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - we want full trace
                elapsed = (time.perf_counter() - start) * 1000
                bound_logger.exception(
                    "Tool execution failed after {:.2f} ms: {}", elapsed, exc
                )
                raise
            elapsed = (time.perf_counter() - start) * 1000
            bound_logger.info(
                "Tool completed in {:.2f} ms | result_preview={}",
                elapsed,
                _safe_repr(result, max_len=300),
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _safe_repr(value: Any, max_len: int = 200) -> str:
    """Return a length-limited repr, tolerant of non-serialisable objects."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001
        text = "<unrepresentable>"
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


# Default application-wide logger.
logger = get_logger("app")
