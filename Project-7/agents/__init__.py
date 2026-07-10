"""
Agentic AI orchestration layer.

This package contains:

    * :mod:`agents.planner_agent` -- the "Intent Understanding" layer that
      turns a free-text user query into a validated
      :class:`models.user_request.TripRequest`, either heuristically or
      (when an OpenAI API key is configured) with LLM-assisted structured
      extraction.
    * :mod:`agents.workflow` -- a LangGraph ``StateGraph`` implementing the
      end-to-end multi-step trip-planning pipeline (flight -> hotel ->
      places -> weather -> budget -> reasoning -> itinerary -> persistence).
    * :mod:`agents.travel_agent` -- a LangChain tool-calling ("ReAct style")
      conversational agent that exposes all five tools directly for
      free-form natural-language chat, for users who configure an OpenAI
      API key.
"""
