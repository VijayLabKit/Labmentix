"""
Conversational Travel Agent.

Provides a free-form, natural-language chat interface to the five travel
planning tools using LangChain's tool-calling agent (built on LangGraph),
which implements a ReAct-style "reason -> act -> observe -> repeat" loop.

This agent **requires** ``settings.OPENAI_API_KEY`` to be configured, since
it relies on an LLM to decide which tools to call and how to phrase the
final answer. For a fully offline, structured alternative, see
:mod:`agents.workflow` (used by the Streamlit "Trip Planner" form), which
implements the same tools as a deterministic LangGraph pipeline.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List

from configs import settings
from models.response_models import AgentRunResult
from tools.budget_tool import budget_estimation_tool
from tools.flight_tool import flight_search_tool
from tools.hotel_tool import hotel_recommendation_tool
from tools.place_tool import places_discovery_tool
from tools.weather_tool import weather_lookup_tool
from utils.llm_provider import get_llm
from utils.logger import get_logger
from utils.validators import global_rate_limiter, sanitize_user_text

logger = get_logger("travel_agent")

TOOLS = [
    flight_search_tool,
    hotel_recommendation_tool,
    places_discovery_tool,
    weather_lookup_tool,
    budget_estimation_tool,
]

SYSTEM_PROMPT = (
    "You are an expert AI travel planning assistant for trips within India, "
    "covering exactly eight cities: Delhi, Mumbai, Bangalore, Chennai, "
    "Kolkata, Hyderabad, Goa and Jaipur. All prices are in INR.\n\n"
    "You have access to five tools: flight_search_tool, "
    "hotel_recommendation_tool, places_discovery_tool, "
    "weather_lookup_tool and budget_estimation_tool. Use them step by step "
    "to gather real data before answering -- never invent flights, hotels, "
    "attractions, prices or weather. Reason about which tool to call next "
    "based on the results of previous tool calls.\n\n"
    "If the user's request is missing key details (source city, destination "
    "city, number of days, number of travellers, or budget), ask a brief "
    "clarifying question before calling tools, unless reasonable defaults "
    "(e.g. 3 days, 1 traveller) would clearly serve the user's intent. "
    "Also, proactively ask clarifying questions regarding interests, weather, duration, "
    "transportation, constraints, seasonality, or traveler type (family/solo/couple/group) "
    "if they would help tailor the itinerary better.\n\n"
    "When you have gathered enough information, present your final response "
    "USING EXACTLY THE FOLLOWING MARKDOWN STRUCTURE:\n"
    "# Executive Summary\n"
    "[Brief overview of the trip, total budget, and chosen travel style]\n\n"
    "# Day-by-Day Itinerary\n"
    "[Detailed daily schedule with morning, afternoon, and evening activities]\n\n"
    "# Attractions\n"
    "[List of selected attractions with reasons for inclusion]\n\n"
    "# Transportation\n"
    "[Flight details and recommended local transport]\n\n"
    "# Budget Breakdown\n"
    "[Detailed cost estimation and category]\n\n"
    "# Food Recommendations\n"
    "[Suggested cuisine and dining options]\n\n"
    "# Weather Notes\n"
    "[Forecast summary and implications]\n\n"
    "# Packing Suggestions\n"
    "[Items to bring based on weather and activities]\n\n"
    "# Safety Information\n"
    "[Local safety tips and emergency contacts]\n\n"
    "# Local Tips\n"
    "[Cultural norms, currency tips, or best practices]\n\n"
    "Be transparent about *why* you chose each option."
)


@lru_cache(maxsize=1)
def build_travel_agent():
    """Build (and cache) the LangChain tool-calling travel agent.

    Returns:
        A compiled LangGraph agent (``CompiledStateGraph``) that can be
        invoked with ``{"messages": [...]}``.

    Raises:
        RuntimeError: If ``settings.OPENAI_API_KEY`` is not configured.
    """
    from langchain.agents import create_agent

    llm = get_llm()
    return create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def _extract_tool_trace(messages: List[Any]) -> List[Dict[str, Any]]:
    """Build a simple ``tool_calls`` trace from a LangGraph message list."""
    trace: List[Dict[str, Any]] = []
    pending: Dict[str, Dict[str, Any]] = {}

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                pending[call["id"]] = {"tool": call["name"], "input": call.get("args", {})}
        msg_type = getattr(message, "type", None)
        if msg_type == "tool":
            call_id = getattr(message, "tool_call_id", None)
            entry = pending.pop(call_id, {"tool": getattr(message, "name", "unknown_tool")})
            content = getattr(message, "content", "")
            entry["output"] = content[:500] if isinstance(content, str) else str(content)[:500]
            trace.append(entry)

    return trace


def run_agent_chat(query: str, session_id: str = "default") -> AgentRunResult:
    """Run a single conversational turn through the tool-calling agent.

    Args:
        query: The user's natural-language message.
        session_id: Client/session identifier, used for rate limiting and
            logging.

    Returns:
        An :class:`models.response_models.AgentRunResult` with the agent's
        final natural-language answer and a trace of every tool call made.
    """
    started = time.perf_counter()

    if not global_rate_limiter.allow(session_id):
        return AgentRunResult(
            success=False,
            final_answer=(
                "You've sent too many messages in a short period. Please "
                "wait a moment and try again."
            ),
            error="rate_limited",
            duration_seconds=time.perf_counter() - started,
        )

    sanitised = sanitize_user_text(query, field_name="agent_query")
    if sanitised.flagged:
        logger.warning(
            "Sanitised chat query for session '{}' flagged patterns: {}",
            session_id,
            sanitised.matched_patterns,
        )

    if not sanitised.text:
        return AgentRunResult(
            success=False,
            final_answer="Please enter a message describing the trip you'd like help with.",
            error="empty_query",
            duration_seconds=time.perf_counter() - started,
        )

    try:
        agent = build_travel_agent()
    except RuntimeError as exc:
        return AgentRunResult(
            success=False,
            final_answer=str(exc),
            error="missing_api_key",
            duration_seconds=time.perf_counter() - started,
        )

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": sanitised.text}]})
        messages = result.get("messages", [])
        final_answer = ""
        for message in reversed(messages):
            if getattr(message, "type", None) == "ai" and getattr(message, "content", None):
                content = message.content
                if isinstance(content, list):
                    text_parts = [
                        part.get("text", "") for part in content if isinstance(part, dict)
                    ]
                    final_answer = "\n".join(text_parts)
                else:
                    final_answer = str(content)
                break

        return AgentRunResult(
            success=True,
            final_answer=final_answer or "I wasn't able to generate a response.",
            tool_calls=_extract_tool_trace(messages),
            duration_seconds=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001 - surface LLM/network errors gracefully
        logger.error("Conversational agent run failed for session '{}': {}", session_id, exc)
        return AgentRunResult(
            success=False,
            final_answer=f"Sorry, the travel agent ran into an error: {exc}",
            error=str(exc),
            duration_seconds=time.perf_counter() - started,
        )
