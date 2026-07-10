"""
Flight Search Tool.

Wraps ``data/flights.json`` with a LangChain ``StructuredTool`` that supports:

    * Filtering by source city
    * Filtering by destination city
    * Filtering by travel date
    * Ranking by "cheapest", "fastest" or "best_value" (price + duration +
      airline rating)

The tool is dataset-size agnostic: it loads the JSON file once per process
(cached) and operates on it with simple list comprehensions, which scale
comfortably from 10 to 10,000+ records for this project's purposes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from configs import settings
from models.response_models import ToolResponse
from services.ranking_engine import rank_flights
from utils.helpers import load_json_dataset, normalise_city, parse_datetime
from utils.logger import log_tool_execution
from utils.validators import ValidationError, validate_city

TOOL_NAME = "flight_search_tool"


@lru_cache(maxsize=1)
def _load_flights() -> List[Dict[str, Any]]:
    """Load and cache the flights dataset for the lifetime of the process."""
    return load_json_dataset(settings.FLIGHTS_FILE)


class FlightSearchInput(BaseModel):
    """Input schema for :func:`search_flights`."""

    source: str = Field(..., description="Departure city, e.g. 'Delhi'")
    destination: str = Field(..., description="Arrival city, e.g. 'Goa'")
    travel_date: Optional[str] = Field(
        default=None,
        description=(
            "Optional travel date in YYYY-MM-DD format. If provided, only "
            "flights departing on this date are considered."
        ),
    )
    criteria: Literal["cheapest", "fastest", "best_value"] = Field(
        default="best_value",
        description=(
            "Ranking criteria: 'cheapest' (lowest price first), 'fastest' "
            "(shortest duration first), or 'best_value' (balanced score of "
            "price, duration and airline rating)."
        ),
    )
    top_k: int = Field(default=3, ge=1, le=20, description="Maximum number of results to return")


@log_tool_execution(TOOL_NAME)
def search_flights(
    source: str,
    destination: str,
    travel_date: Optional[str] = None,
    criteria: str = "best_value",
    top_k: int = 3,
) -> str:
    """Search and rank flights between two cities.

    Args:
        source: Departure city name.
        destination: Arrival city name.
        travel_date: Optional ``YYYY-MM-DD`` date filter.
        criteria: ``"cheapest"``, ``"fastest"`` or ``"best_value"``.
        top_k: Maximum number of results to return.

    Returns:
        A JSON string encoding a :class:`models.response_models.ToolResponse`
        whose ``data`` field is a list of ranked flight dictionaries (each
        including ``duration_minutes``, ``duration_label``,
        ``airline_rating`` and, for ``best_value``, ``value_score``), or an
        ``error`` field describing what went wrong.
    """
    try:
        source_city = validate_city(source, "source")
        destination_city = validate_city(destination, "destination")
    except ValidationError as exc:
        return ToolResponse.fail(TOOL_NAME, error=str(exc)).to_json()

    flights = _load_flights()

    matches = [
        flight
        for flight in flights
        if normalise_city(flight["from"]) == source_city
        and normalise_city(flight["to"]) == destination_city
    ]

    if travel_date:
        try:
            target_date = parse_datetime(f"{travel_date}T00:00:00").date()
        except ValueError:
            return ToolResponse.fail(
                TOOL_NAME, error=f"Invalid travel_date '{travel_date}'. Expected YYYY-MM-DD."
            ).to_json()
        matches = [
            flight
            for flight in matches
            if parse_datetime(flight["departure_time"]).date() == target_date
        ]

    if not matches:
        return ToolResponse.ok(
            TOOL_NAME,
            data=[],
            message=(
                f"No flights found from {source_city} to {destination_city}"
                + (f" on {travel_date}" if travel_date else "")
                + ". Try a different date or route."
            ),
        ).to_json()

    ranked = rank_flights(matches, criteria=criteria)  # type: ignore[arg-type]
    top_results = ranked[:top_k]

    return ToolResponse.ok(
        TOOL_NAME,
        data=top_results,
        message=(
            f"Found {len(matches)} flight(s) from {source_city} to {destination_city}; "
            f"returning top {len(top_results)} by '{criteria}'."
        ),
    ).to_json()


flight_search_tool = StructuredTool.from_function(
    func=search_flights,
    name=TOOL_NAME,
    description=(
        "Search flights between a source and destination city from the flights "
        "dataset. Supports ranking by 'cheapest', 'fastest' or 'best_value' "
        "(a balanced score of price, flight duration and airline rating). "
        "Optionally filter by an exact travel date (YYYY-MM-DD). Returns a "
        "JSON ToolResponse containing the ranked flights."
    ),
    args_schema=FlightSearchInput,
)
