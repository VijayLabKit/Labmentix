"""
Generic response envelope models.

Every LangChain tool returns its payload wrapped in a :class:`ToolResponse`
(serialised to a JSON string, since LangChain tools must return strings or
simple structures). This gives a consistent ``status`` / ``data`` / ``error``
/ ``message`` contract that the agent's reasoning layer and the Streamlit UI
can rely on regardless of which tool produced the response.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ToolResponse(BaseModel, Generic[T]):
    """Standard envelope returned by every tool in :mod:`tools`."""

    status: str = Field(description="'success' or 'error'")
    tool_name: str
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def ok(cls, tool_name: str, data: Any, message: str = "") -> "ToolResponse":
        """Build a success response."""
        return cls(status="success", tool_name=tool_name, data=data, message=message)

    @classmethod
    def fail(cls, tool_name: str, error: str, message: str = "") -> "ToolResponse":
        """Build an error response."""
        return cls(status="error", tool_name=tool_name, data=None, error=error, message=message)

    def to_json(self) -> str:
        """Serialise to a compact JSON string for LangChain tool output."""
        return self.model_dump_json()


class AgentRunResult(BaseModel):
    """Result returned to the UI after a full agent run."""

    success: bool
    final_answer: str
    itinerary_json: Optional[dict] = None
    tool_calls: list = Field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


def dumps(value: Any) -> str:
    """Convenience JSON dumper that handles dataclasses/Pydantic models."""

    def _default(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    return json.dumps(value, default=_default, ensure_ascii=False)
