"""
Hotel Recommendation Tool.

Wraps ``data/hotels.json`` with a LangChain ``StructuredTool`` that supports:

    * Filtering by city
    * Filtering by maximum budget (price per night)
    * Filtering by minimum star rating
    * Sorting by price, rating, or a combined "best_value" score
    * Optional semantic re-ranking via the FAISS-backed
      :class:`services.recommendation_engine.RecommendationEngine` when the
      caller supplies a free-text preference (e.g. "quiet hotel with a pool
      and spa").
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from configs import settings
from models.response_models import ToolResponse
from services.ranking_engine import rank_hotels
from services.recommendation_engine import RecommendationEngine
from utils.helpers import load_json_dataset, normalise_city
from utils.logger import log_tool_execution
from utils.validators import ValidationError, validate_city

TOOL_NAME = "hotel_recommendation_tool"


@lru_cache(maxsize=1)
def _load_hotels() -> List[Dict[str, Any]]:
    """Load and cache the hotels dataset for the lifetime of the process."""
    return load_json_dataset(settings.HOTELS_FILE)


@lru_cache(maxsize=1)
def _load_places_for_engine() -> List[Dict[str, Any]]:
    """Load the places dataset (needed to construct the shared FAISS engine)."""
    return load_json_dataset(settings.PLACES_FILE)


@lru_cache(maxsize=1)
def _get_recommendation_engine() -> RecommendationEngine:
    """Build (and cache) the shared semantic recommendation engine."""
    return RecommendationEngine(hotels=_load_hotels(), places=_load_places_for_engine())


class HotelSearchInput(BaseModel):
    """Input schema for :func:`search_hotels`."""

    city: str = Field(..., description="City to search hotels in, e.g. 'Goa'")
    max_price_per_night: Optional[float] = Field(
        default=None, description="Optional maximum nightly price filter (INR)"
    )
    min_stars: Optional[int] = Field(
        default=None, ge=1, le=5, description="Optional minimum star rating filter (1-5)"
    )
    sort_by: Literal["price", "rating", "best_value"] = Field(
        default="best_value",
        description=(
            "Sort order: 'price' (cheapest first), 'rating' (highest stars "
            "first), or 'best_value' (balanced score of price and rating)."
        ),
    )
    preference_text: Optional[str] = Field(
        default=None,
        description=(
            "Optional free-text description of what the traveller wants "
            "(e.g. 'hotel with pool and spa for a relaxing stay'). When "
            "provided, results are semantically re-ranked using a FAISS "
            "vector search over hotel descriptions."
        ),
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of results to return")


@log_tool_execution(TOOL_NAME)
def search_hotels(
    city: str,
    max_price_per_night: Optional[float] = None,
    min_stars: Optional[int] = None,
    sort_by: str = "best_value",
    preference_text: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """Search, filter and rank hotels in a given city.

    Args:
        city: City to search hotels in.
        max_price_per_night: Optional maximum nightly price (INR).
        min_stars: Optional minimum star rating (1-5).
        sort_by: ``"price"``, ``"rating"`` or ``"best_value"``.
        preference_text: Optional free-text preference used for semantic
            (FAISS) re-ranking.
        top_k: Maximum number of results to return.

    Returns:
        A JSON string encoding a :class:`models.response_models.ToolResponse`
        whose ``data`` field is a list of ranked hotel dictionaries.
    """
    try:
        city_name = validate_city(city, "city")
    except ValidationError as exc:
        return ToolResponse.fail(TOOL_NAME, error=str(exc)).to_json()

    hotels = _load_hotels()
    matches = [hotel for hotel in hotels if normalise_city(hotel["city"]) == city_name]

    if max_price_per_night is not None:
        matches = [h for h in matches if h["price_per_night"] <= max_price_per_night]

    if min_stars is not None:
        matches = [h for h in matches if h["stars"] >= min_stars]

    if not matches:
        return ToolResponse.ok(
            TOOL_NAME,
            data=[],
            message=f"No hotels found in {city_name} matching the given filters.",
        ).to_json()

    ranked = rank_hotels(matches, criteria=sort_by)  # type: ignore[arg-type]

    if preference_text and preference_text.strip():
        engine = _get_recommendation_engine()
        semantic_results = engine.recommend_hotels(preference_text, top_k=len(ranked))
        semantic_scores = {h["hotel_id"]: h.get("semantic_score") for h in semantic_results}
        for hotel in ranked:
            hotel["semantic_score"] = semantic_scores.get(hotel["hotel_id"], 0.0)
        ranked = sorted(ranked, key=lambda h: h.get("semantic_score", 0.0), reverse=True)

    top_results = ranked[:top_k]

    return ToolResponse.ok(
        TOOL_NAME,
        data=top_results,
        message=(
            f"Found {len(matches)} hotel(s) in {city_name} matching filters; "
            f"returning top {len(top_results)} sorted by '{sort_by}'"
            + (" with semantic re-ranking." if preference_text else ".")
        ),
    ).to_json()


hotel_recommendation_tool = StructuredTool.from_function(
    func=search_hotels,
    name=TOOL_NAME,
    description=(
        "Search and rank hotels in a given city from the hotels dataset. "
        "Supports filtering by maximum nightly price and minimum star "
        "rating, sorting by 'price', 'rating' or 'best_value', and optional "
        "semantic re-ranking via a free-text 'preference_text' (e.g. "
        "'hotel with pool and spa'). Returns a JSON ToolResponse containing "
        "the ranked hotels."
    ),
    args_schema=HotelSearchInput,
)
