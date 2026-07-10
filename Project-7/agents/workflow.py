"""
LangGraph multi-step trip-planning workflow.

This module implements the core "agentic" pipeline described in the
project brief as a `LangGraph <https://github.com/langchain-ai/langgraph>`_
``StateGraph``::

    User Query -> Intent Understanding (agents.planner_agent)
                -> Flight Search Tool
                -> Hotel Recommendation Tool
                -> Places Discovery Tool
                -> Weather Lookup Tool
                -> Budget Estimation Tool (with one budget-aware re-selection)
                -> Reasoning Layer
                -> Itinerary Generator
                -> Persistence
                -> Final Response (JSON + human-readable)

Each node is a small, independently testable function operating on a
:class:`TravelPlanState` ``TypedDict``. The graph is deterministic and
requires **no** external API access to run end-to-end -- the Reasoning
Layer falls back to template-based explanations when no OpenAI API key is
configured, and the Weather node degrades gracefully (empty forecast with a
warning) if Open-Meteo is unreachable or the dates are out of range.

The public entry point is :func:`run_trip_workflow`.
"""

from __future__ import annotations

import json
import time
from datetime import date as date_type
from functools import lru_cache
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from configs import settings
from database.database import get_database
from models.itinerary import Itinerary
from models.response_models import AgentRunResult
from models.user_request import TripRequest
from services.itinerary_builder import build_itinerary
from tools.budget_tool import budget_estimation_tool
from tools.flight_tool import flight_search_tool
from tools.hotel_tool import hotel_recommendation_tool
from tools.place_tool import places_discovery_tool
from tools.weather_tool import weather_lookup_tool
from utils.logger import get_logger
from utils.validators import global_rate_limiter

logger = get_logger("workflow")


# ---------------------------------------------------------------------------
# Travel-style based tool parameter selection
# ---------------------------------------------------------------------------

FLIGHT_CRITERIA_BY_STYLE: Dict[str, str] = {
    "Family": "best_value",
    "Adventure": "best_value",
    "Luxury": "fastest",
    "Backpacker": "cheapest",
}

HOTEL_SORT_BY_STYLE: Dict[str, str] = {
    "Family": "best_value",
    "Adventure": "best_value",
    "Luxury": "rating",
    "Backpacker": "price",
}

HOTEL_PREFERENCE_BY_STYLE: Dict[str, str] = {
    "Family": "family friendly hotel with breakfast, parking and pool for a comfortable stay",
    "Adventure": "hotel with gym and a convenient location as a base for adventure activities",
    "Luxury": "luxury hotel with spa, pool, breakfast and premium amenities",
    "Backpacker": "budget friendly hotel with free wifi and basic amenities",
}

PLACE_CATEGORY_BY_STYLE: Dict[str, str] = {
    "Family": "Family",
    "Adventure": "Adventure",
    "Luxury": "Relaxation",
    "Backpacker": "Cultural",
}

PLACE_PREFERENCE_BY_STYLE: Dict[str, str] = {
    "Family": "family friendly parks, lakes and markets",
    "Adventure": "adventure activities, beaches, forts and outdoor exploration",
    "Luxury": "relaxing scenic spots, beaches and premium experiences",
    "Backpacker": "historical and cultural sites, museums, temples and forts",
}


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------


class TravelPlanState(TypedDict, total=False):
    """Shared state threaded through every node of the workflow graph."""

    trip_request: Dict[str, Any]
    session_id: str
    flights: List[Dict[str, Any]]
    selected_flight: Optional[Dict[str, Any]]
    hotels: List[Dict[str, Any]]
    selected_hotel: Optional[Dict[str, Any]]
    places: List[Dict[str, Any]]
    weather: List[Dict[str, Any]]
    budget: Optional[Dict[str, Any]]
    reasoning: Dict[str, str]
    itinerary: Optional[Dict[str, Any]]
    itinerary_markdown: Optional[str]
    tool_calls: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke_tool(tool: Any, tool_input: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Invoke a LangChain ``StructuredTool`` and return ``(parsed, record)``.

    ``parsed`` is the decoded :class:`models.response_models.ToolResponse`
    dict. ``record`` is a small trace entry suitable for
    :class:`models.response_models.AgentRunResult.tool_calls` and for
    persisting via :meth:`database.database.TravelDatabase.log_tool_call`.
    """
    start = time.perf_counter()
    try:
        raw = tool.invoke(tool_input)
        duration_ms = (time.perf_counter() - start) * 1000
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - surface as a failed tool call
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("Tool '{}' raised an exception: {}", getattr(tool, "name", tool), exc)
        parsed = {
            "status": "error",
            "tool_name": getattr(tool, "name", "unknown_tool"),
            "data": None,
            "error": str(exc),
            "message": None,
        }

    record = {
        "tool": parsed.get("tool_name", getattr(tool, "name", "unknown_tool")),
        "input": tool_input,
        "status": parsed.get("status"),
        "message": parsed.get("message") or parsed.get("error"),
        "duration_ms": round(duration_ms, 2),
    }
    return parsed, record


def _append(state: TravelPlanState, key: str, value: Any) -> List[Any]:
    """Return a new list with ``value`` appended to ``state.get(key, [])``."""
    return list(state.get(key, [])) + [value]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def node_flight_search(state: TravelPlanState) -> Dict[str, Any]:
    """Search for flights between the trip's source and destination cities."""
    trip = state["trip_request"]
    criteria = FLIGHT_CRITERIA_BY_STYLE.get(trip["travel_style"], "best_value")

    parsed, record = _invoke_tool(
        flight_search_tool,
        {
            "source": trip["source_city"],
            "destination": trip["destination_city"],
            "travel_date": None,
            "criteria": criteria,
            "top_k": 5,
        },
    )

    flights: List[Dict[str, Any]] = []
    selected_flight: Optional[Dict[str, Any]] = None
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    if parsed.get("status") == "error":
        errors.append(f"flight_search_tool: {parsed.get('error')}")
    else:
        flights = parsed.get("data") or []
        if flights:
            selected_flight = flights[0]
        else:
            warnings.append(
                f"No flights found from {trip['source_city']} to "
                f"{trip['destination_city']} in the dataset; the itinerary "
                "will proceed without flight details."
            )

    return {
        "flights": flights,
        "selected_flight": selected_flight,
        "tool_calls": _append(state, "tool_calls", record),
        "warnings": warnings,
        "errors": errors,
    }


def node_hotel_search(state: TravelPlanState) -> Dict[str, Any]:
    """Search for and recommend a hotel in the destination city."""
    trip = state["trip_request"]
    style = trip["travel_style"]

    parsed, record = _invoke_tool(
        hotel_recommendation_tool,
        {
            "city": trip["destination_city"],
            "sort_by": HOTEL_SORT_BY_STYLE.get(style, "best_value"),
            "preference_text": HOTEL_PREFERENCE_BY_STYLE.get(style),
            "top_k": 5,
        },
    )

    hotels: List[Dict[str, Any]] = []
    selected_hotel: Optional[Dict[str, Any]] = None
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    if parsed.get("status") == "error":
        errors.append(f"hotel_recommendation_tool: {parsed.get('error')}")
    else:
        hotels = parsed.get("data") or []
        if hotels:
            selected_hotel = hotels[0]
        else:
            warnings.append(
                f"No hotels found in {trip['destination_city']}; the itinerary "
                "will proceed without hotel details."
            )

    return {
        "hotels": hotels,
        "selected_hotel": selected_hotel,
        "tool_calls": _append(state, "tool_calls", record),
        "warnings": warnings,
        "errors": errors,
    }


def node_places_search(state: TravelPlanState) -> Dict[str, Any]:
    """Discover attractions in the destination city for the day plans.

    Attractions matching the travel style's preferred category are
    prioritised, but the result list is always topped up with other
    highly-rated attractions in the city so that day plans have enough
    variety even when a category is sparsely represented in the dataset.
    """
    trip = state["trip_request"]
    style = trip["travel_style"]
    desired = max(len(settings.SLOTS_PER_DAY) * trip["num_days"], 6)
    top_k = min(desired, 40)
    category = PLACE_CATEGORY_BY_STYLE.get(style)

    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))
    tool_calls = list(state.get("tool_calls", []))

    category_places: List[Dict[str, Any]] = []
    if category:
        parsed, record = _invoke_tool(
            places_discovery_tool,
            {
                "city": trip["destination_city"],
                "category": category,
                "preference_text": PLACE_PREFERENCE_BY_STYLE.get(style),
                "top_k": top_k,
            },
        )
        tool_calls.append(record)
        if parsed.get("status") == "error":
            errors.append(f"places_discovery_tool: {parsed.get('error')}")
        else:
            category_places = parsed.get("data") or []

    # Always fetch a broader pool (no category filter) to top up variety if
    # the category-specific result set is small or empty.
    places: List[Dict[str, Any]] = list(category_places)
    if len(places) < top_k and not errors:
        parsed, record = _invoke_tool(
            places_discovery_tool,
            {
                "city": trip["destination_city"],
                "category": None,
                "preference_text": PLACE_PREFERENCE_BY_STYLE.get(style),
                "top_k": top_k,
            },
        )
        tool_calls.append(record)
        if parsed.get("status") == "error":
            errors.append(f"places_discovery_tool: {parsed.get('error')}")
        else:
            seen_ids = {p["place_id"] for p in places}
            for place in parsed.get("data") or []:
                if place["place_id"] not in seen_ids:
                    places.append(place)
                    seen_ids.add(place["place_id"])
            places = places[:top_k]

            if category and not category_places:
                warnings.append(
                    f"No '{category}'-category attractions found in "
                    f"{trip['destination_city']}; showing top-rated "
                    "attractions across all categories instead."
                )
            elif category and len(category_places) < len(settings.SLOTS_PER_DAY):
                warnings.append(
                    f"Only {len(category_places)} '{category}'-category "
                    f"attraction(s) found in {trip['destination_city']}; "
                    "the day plans are topped up with other highly-rated "
                    "attractions for variety."
                )

    if not places and not errors:
        warnings.append(
            f"No attractions found in {trip['destination_city']}; day plans "
            "will suggest free time / self-exploration."
        )

    return {
        "places": places,
        "tool_calls": tool_calls,
        "warnings": warnings,
        "errors": errors,
    }


def node_weather(state: TravelPlanState) -> Dict[str, Any]:
    """Fetch a daily weather forecast for the destination city."""
    trip = state["trip_request"]
    forecast_days = min(trip["num_days"], settings.WEATHER_FORECAST_DAYS)

    parsed, record = _invoke_tool(
        weather_lookup_tool,
        {
            "city": trip["destination_city"],
            "start_date": trip["start_date"],
            "num_days": forecast_days,
        },
    )

    weather: List[Dict[str, Any]] = []
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    if parsed.get("status") == "error":
        warnings.append(
            f"Weather forecast unavailable ({parsed.get('error')}); day plans "
            "will be shown without weather information."
        )
    else:
        weather = parsed.get("data") or []
        if not weather:
            warnings.append(
                "Weather forecast returned no data for the requested dates "
                "(they may be too far in the future); day plans will be "
                "shown without weather information."
            )
        if trip["num_days"] > forecast_days:
            warnings.append(
                f"Weather forecasts are only available up to "
                f"{settings.WEATHER_FORECAST_DAYS} days ahead; the remaining "
                f"{trip['num_days'] - forecast_days} day(s) of the trip will "
                "be shown without weather information."
            )

    return {
        "weather": weather,
        "tool_calls": _append(state, "tool_calls", record),
        "warnings": warnings,
        "errors": errors,
    }


def node_budget(state: TravelPlanState) -> Dict[str, Any]:
    """Estimate the trip budget, re-selecting cheaper options if over budget."""
    trip = state["trip_request"]
    selected_flight = state.get("selected_flight")
    selected_hotel = state.get("selected_hotel")
    flights = state.get("flights", [])
    hotels = state.get("hotels", [])
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))
    tool_calls = list(state.get("tool_calls", []))

    flight_price = float(selected_flight["price"]) if selected_flight else 0.0
    hotel_price = float(selected_hotel["price_per_night"]) if selected_hotel else 0.0

    parsed, record = _invoke_tool(
        budget_estimation_tool,
        {
            "flight_price": flight_price,
            "hotel_price_per_night": hotel_price,
            "num_days": trip["num_days"],
            "num_travellers": trip["num_travellers"],
        },
    )
    tool_calls.append(record)

    if parsed.get("status") == "error":
        errors.append(f"budget_estimation_tool: {parsed.get('error')}")
        return {"budget": None, "tool_calls": tool_calls, "warnings": warnings, "errors": errors}

    budget = parsed.get("data")

    # Budget-aware re-selection: if the estimated total cost exceeds the
    # traveller's stated budget, swap in the cheapest available flight and
    # hotel and recompute once.
    if budget and budget["total_cost"] > trip["budget"] and (flights or hotels):
        cheaper_flight = min(flights, key=lambda f: f["price"]) if flights else selected_flight
        cheaper_hotel = (
            min(hotels, key=lambda h: h["price_per_night"]) if hotels else selected_hotel
        )

        changed = (cheaper_flight != selected_flight) or (cheaper_hotel != selected_hotel)
        if changed:
            retry_parsed, retry_record = _invoke_tool(
                budget_estimation_tool,
                {
                    "flight_price": float(cheaper_flight["price"]) if cheaper_flight else 0.0,
                    "hotel_price_per_night": (
                        float(cheaper_hotel["price_per_night"]) if cheaper_hotel else 0.0
                    ),
                    "num_days": trip["num_days"],
                    "num_travellers": trip["num_travellers"],
                },
            )
            tool_calls.append(retry_record)
            if retry_parsed.get("status") == "success":
                new_budget = retry_parsed.get("data")
                if new_budget and new_budget["total_cost"] < budget["total_cost"]:
                    warnings.append(
                        "The originally recommended flight/hotel exceeded your "
                        "stated budget, so a more economical flight and/or "
                        "hotel was selected instead."
                    )
                    budget = new_budget
                    selected_flight = cheaper_flight
                    selected_hotel = cheaper_hotel

    if budget and budget["total_cost"] > trip["budget"]:
        errors.append(
            f"Your stated budget of INR {trip['budget']:,.0f} is too low for a "
            f"{trip['num_days']}-day trip for {trip['num_travellers']} "
            f"traveller(s) to {trip['destination_city']}. The cheapest estimated "
            f"cost we found is INR {budget['total_cost']:,.0f}. "
            "Please increase your budget, or consider a shorter trip, fewer travellers, "
            "or another destination."
        )

    return {
        "budget": budget,
        "selected_flight": selected_flight,
        "selected_hotel": selected_hotel,
        "tool_calls": tool_calls,
        "warnings": warnings,
        "errors": errors,
    }


class _ReasoningOutput(BaseModel):
    """Structured schema requested from the LLM in :func:`node_reasoning`."""

    flight_reasoning: str = Field(description="1-2 sentences explaining the flight choice")
    hotel_reasoning: str = Field(description="1-2 sentences explaining the hotel choice")
    attractions_reasoning: str = Field(
        description="1-2 sentences explaining the chosen attractions/categories"
    )
    itinerary_ordering_reasoning: str = Field(
        description="1-2 sentences explaining how the days were structured"
    )


def _template_reasoning(state: TravelPlanState) -> Dict[str, str]:
    """Build a deterministic, template-based reasoning trace.

    This requires no LLM and is always available, ensuring the application
    is fully functional offline.
    """
    trip = state["trip_request"]
    style = trip["travel_style"]
    selected_flight = state.get("selected_flight")
    selected_hotel = state.get("selected_hotel")
    places = state.get("places", [])
    budget = state.get("budget")
    criteria = FLIGHT_CRITERIA_BY_STYLE.get(style, "best_value")
    hotel_sort = HOTEL_SORT_BY_STYLE.get(style, "best_value")

    if selected_flight:
        flight_reasoning = (
            f"Selected {selected_flight['airline']} flight "
            f"{selected_flight['flight_id']} (INR {selected_flight['price']:,.0f}, "
            f"{selected_flight.get('duration_label', 'duration n/a')}, airline "
            f"rating {selected_flight.get('airline_rating', 'n/a')}/5) using the "
            f"'{criteria}' ranking criteria, which best matches a "
            f"'{style}' travel style."
        )
    else:
        flight_reasoning = (
            f"No flight could be selected for the {trip['source_city']} -> "
            f"{trip['destination_city']} route in the available dataset."
        )

    if selected_hotel:
        amenities = ", ".join(selected_hotel.get("amenities", [])) or "standard amenities"
        hotel_reasoning = (
            f"Selected {selected_hotel['name']} in {selected_hotel['city']} "
            f"({selected_hotel['stars']}-star, INR {selected_hotel['price_per_night']:,.0f}/night, "
            f"amenities: {amenities}) using the '{hotel_sort}' sort order, "
            f"suited to a '{style}' travel style."
        )
    else:
        hotel_reasoning = f"No hotel could be selected in {trip['destination_city']}."

    if places:
        sample_names = ", ".join(p["name"] for p in places[:3])
        categories = PLACE_CATEGORY_BY_STYLE.get(style, "Cultural")
        attractions_reasoning = (
            f"Chose {len(places)} attraction(s) in {trip['destination_city']} "
            f"prioritising the '{categories}' category for a '{style}' trip, "
            f"including {sample_names}, ranked by visitor rating."
        )
    else:
        attractions_reasoning = (
            f"No attractions were found in {trip['destination_city']} in the "
            "dataset, so day plans default to free time / self-exploration."
        )

    itinerary_ordering_reasoning = (
        f"Attractions are cycled across {trip['num_days']} day(s) and the "
        f"{', '.join(settings.SLOTS_PER_DAY)} slots so the itinerary stays "
        "balanced even if there are fewer unique attractions than slots; "
        "each day also shows the matching weather forecast where available."
    )
    if budget:
        itinerary_ordering_reasoning += (
            f" The overall plan falls into the '{budget['budget_category']}' "
            f"budget category (INR {budget['total_cost']:,.0f} total, "
            f"INR {budget['per_traveller_cost']:,.0f} per traveller)."
        )

    return {
        "flight": flight_reasoning,
        "hotel": hotel_reasoning,
        "attractions": attractions_reasoning,
        "itinerary_ordering": itinerary_ordering_reasoning,
    }


def _llm_reasoning(state: TravelPlanState, fallback: Dict[str, str]) -> Dict[str, str]:
    """Attempt to enrich the reasoning trace using an LLM.

    Falls back to ``fallback`` (the template-based reasoning) on any error
    or if no API key is configured.
    """
    if not settings.GEMINI_API_KEY:
        return fallback

    try:
        from utils.llm_provider import get_llm

        trip = state["trip_request"]
        llm = get_llm()
        structured_llm = llm.with_structured_output(_ReasoningOutput)

        prompt = (
            "You are the reasoning layer of a travel-planning agent. Given the "
            "trip request and the selections below, write concise (1-2 "
            "sentence) explanations for each of: the flight choice, the hotel "
            "choice, the attraction selection, and how the day-by-day "
            "itinerary is structured. Be specific, citing prices/ratings "
            "where relevant.\n\n"
            f"Trip request: {json.dumps(trip)}\n"
            f"Selected flight: {json.dumps(state.get('selected_flight'))}\n"
            f"Selected hotel: {json.dumps(state.get('selected_hotel'))}\n"
            f"Attractions: {json.dumps(state.get('places', [])[:5])}\n"
            f"Budget breakdown: {json.dumps(state.get('budget'))}\n"
        )
        result = structured_llm.invoke(prompt)
        if not isinstance(result, _ReasoningOutput):
            result = _ReasoningOutput(**result)  # type: ignore[arg-type]
        return {
            "flight": result.flight_reasoning,
            "hotel": result.hotel_reasoning,
            "attractions": result.attractions_reasoning,
            "itinerary_ordering": result.itinerary_ordering_reasoning,
        }
    except Exception as exc:  # noqa: BLE001 - LLM reasoning is best-effort
        logger.warning("LLM-assisted reasoning failed, using template reasoning: {}", exc)
        return fallback


def node_reasoning(state: TravelPlanState) -> Dict[str, Any]:
    """Produce a transparent reasoning trace for every major decision."""
    template = _template_reasoning(state)
    reasoning = _llm_reasoning(state, template)
    return {"reasoning": reasoning}


def node_itinerary(state: TravelPlanState) -> Dict[str, Any]:
    """Assemble the final structured itinerary and its Markdown rendering."""
    trip = state["trip_request"]
    errors = list(state.get("errors", []))

    if errors:
        return {"itinerary": None, "itinerary_markdown": None, "errors": errors}

    budget = state.get("budget") or {
        "flight_cost": 0.0,
        "hotel_cost": 0.0,
        "food_cost": 0.0,
        "local_transport_cost": 0.0,
        "miscellaneous_cost": 0.0,
        "total_cost": 0.0,
        "daily_budget": 0.0,
        "budget_category": "Budget",
        "currency": "INR",
        "num_travellers": trip["num_travellers"],
        "per_traveller_cost": 0.0,
    }

    try:
        itinerary: Itinerary = build_itinerary(
            source_city=trip["source_city"],
            destination_city=trip["destination_city"],
            start_date=date_type.fromisoformat(trip["start_date"]),
            num_days=trip["num_days"],
            travel_style=trip["travel_style"],
            selected_flight=state.get("selected_flight"),
            selected_hotel=state.get("selected_hotel"),
            ranked_places=state.get("places", []),
            weather_forecast=state.get("weather", []),
            budget_breakdown=budget,
            reasoning=state.get("reasoning", {}),
        )
    except Exception as exc:  # noqa: BLE001 - surface build errors without crashing
        logger.error("Failed to build itinerary: {}", exc)
        errors.append(f"itinerary_builder: {exc}")
        return {"itinerary": None, "itinerary_markdown": None, "errors": errors}

    return {
        "itinerary": itinerary.model_dump(),
        "itinerary_markdown": itinerary.to_markdown(),
        "errors": errors,
    }


def node_persist(state: TravelPlanState) -> Dict[str, Any]:
    """Persist the request, itinerary, selections and tool call logs to SQLite."""
    trip = state["trip_request"]
    session_id = state.get("session_id", "default")
    warnings = list(state.get("warnings", []))

    try:
        db = get_database()
        budget = state.get("budget") or {}
        query_id = db.insert_user_query(
            session_id=session_id,
            source_city=trip["source_city"],
            destination_city=trip["destination_city"],
            start_date=trip["start_date"],
            num_days=trip["num_days"],
            budget=trip["budget"],
            travel_style=trip["travel_style"],
            num_travellers=trip["num_travellers"],
            raw_query=trip.get("raw_query"),
        )

        itinerary = state.get("itinerary")
        if itinerary:
            db.insert_itinerary(
                query_id=query_id,
                trip_title=itinerary["trip_title"],
                itinerary=itinerary,
                total_cost=budget.get("total_cost"),
                budget_category=budget.get("budget_category"),
            )

        selected_flight = state.get("selected_flight")
        if selected_flight:
            db.insert_flight_selection(
                query_id=query_id,
                flight_id=selected_flight["flight_id"],
                airline=selected_flight["airline"],
                source_city=trip["source_city"],
                destination_city=trip["destination_city"],
                price=selected_flight["price"],
                duration_minutes=selected_flight.get("duration_minutes", 0),
                selection_reason=state.get("reasoning", {}).get("flight"),
            )

        selected_hotel = state.get("selected_hotel")
        if selected_hotel:
            db.insert_hotel_selection(
                query_id=query_id,
                hotel_id=selected_hotel["hotel_id"],
                hotel_name=selected_hotel["name"],
                city=selected_hotel["city"],
                stars=selected_hotel["stars"],
                price_per_night=selected_hotel["price_per_night"],
                selection_reason=state.get("reasoning", {}).get("hotel"),
            )

        for call in state.get("tool_calls", []):
            db.log_tool_call(
                session_id=session_id,
                tool_name=call["tool"],
                input_payload=json.dumps(call.get("input"), default=str),
                output_status=call.get("status", "unknown"),
                duration_ms=call.get("duration_ms"),
                error_message=call.get("message") if call.get("status") == "error" else None,
            )
    except Exception as exc:  # noqa: BLE001 - persistence failures must not break the response
        logger.error("Failed to persist trip planning results: {}", exc)
        warnings.append(f"Could not save this trip to history: {exc}")

    return {"warnings": warnings}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_graph():
    """Build and compile the LangGraph workflow (cached for the process)."""
    graph = StateGraph(TravelPlanState)

    graph.add_node("flight_search", node_flight_search)
    graph.add_node("hotel_search", node_hotel_search)
    graph.add_node("places_search", node_places_search)
    graph.add_node("weather", node_weather)
    graph.add_node("budget", node_budget)
    graph.add_node("reasoning", node_reasoning)
    graph.add_node("itinerary", node_itinerary)
    graph.add_node("persist", node_persist)

    graph.set_entry_point("flight_search")
    graph.add_edge("flight_search", "hotel_search")
    graph.add_edge("hotel_search", "places_search")
    graph.add_edge("places_search", "weather")
    graph.add_edge("weather", "budget")
    graph.add_edge("budget", "reasoning")
    graph.add_edge("reasoning", "itinerary")
    graph.add_edge("itinerary", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


def run_trip_workflow(trip_request: TripRequest, session_id: str = "default") -> AgentRunResult:
    """Run the full multi-step trip-planning workflow for a validated request.

    Args:
        trip_request: A validated :class:`models.user_request.TripRequest`.
        session_id: Client/session identifier, used for rate limiting,
            persistence and logging.

    Returns:
        An :class:`models.response_models.AgentRunResult` containing the
        human-readable itinerary (Markdown), the structured itinerary JSON,
        a trace of every tool call made, and any warnings/errors.
    """
    started = time.perf_counter()

    if not global_rate_limiter.allow(session_id):
        return AgentRunResult(
            success=False,
            final_answer=(
                "You've made too many planning requests in a short period. "
                "Please wait a moment and try again."
            ),
            error="rate_limited",
            duration_seconds=time.perf_counter() - started,
        )

    initial_state: TravelPlanState = {
        "trip_request": trip_request.model_dump(mode="json"),
        "session_id": session_id,
        "tool_calls": [],
        "warnings": [],
        "errors": [],
    }

    try:
        graph = _build_graph()
        final_state: TravelPlanState = graph.invoke(initial_state)  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001 - the workflow itself must not crash the UI
        logger.error("Trip planning workflow failed: {}", exc)
        return AgentRunResult(
            success=False,
            final_answer=f"Sorry, something went wrong while planning your trip: {exc}",
            error=str(exc),
            duration_seconds=time.perf_counter() - started,
        )

    itinerary = final_state.get("itinerary")
    markdown = final_state.get("itinerary_markdown")
    errors = final_state.get("errors", [])
    warnings = final_state.get("warnings", [])

    if itinerary and markdown:
        final_answer = markdown
        if warnings:
            final_answer += "\n\n---\n**Notes:**\n" + "\n".join(f"- {w}" for w in warnings)
        success = True
    else:
        final_answer = "Sorry, we couldn't build an itinerary for this request."
        if errors:
            final_answer += " " + "; ".join(errors)
        success = False

    return AgentRunResult(
        success=success,
        final_answer=final_answer,
        itinerary_json=itinerary,
        tool_calls=final_state.get("tool_calls", []),
        error="; ".join(errors) if errors else None,
        duration_seconds=time.perf_counter() - started,
    )
