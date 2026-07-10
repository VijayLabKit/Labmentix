"""
Places Discovery Tool.

Wraps ``data/places.json`` with a LangChain ``StructuredTool`` that supports:

    * Ranking attractions by rating
    * Category filtering (Family / Adventure / Historical / Cultural /
      Relaxation), derived from each place's raw ``type`` field via
      :data:`configs.settings.PLACE_TYPE_CATEGORY_MAP`
    * Optional semantic re-ranking via the FAISS-backed
      :class:`services.recommendation_engine.RecommendationEngine`
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from configs import settings
from models.response_models import ToolResponse
from services.ranking_engine import rank_places
from services.recommendation_engine import RecommendationEngine
from utils.helpers import categorise_place, load_json_dataset, normalise_city
from utils.logger import log_tool_execution
from utils.validators import ValidationError, validate_city

TOOL_NAME = "places_discovery_tool"

VALID_CATEGORIES = {"family", "adventure", "historical", "cultural", "relaxation"}


@lru_cache(maxsize=1)
def _load_places() -> List[Dict[str, Any]]:
    """Load and cache the places dataset for the lifetime of the process."""
    return load_json_dataset(settings.PLACES_FILE)


@lru_cache(maxsize=1)
def _load_hotels_for_engine() -> List[Dict[str, Any]]:
    """Load the hotels dataset (needed to construct the shared FAISS engine)."""
    return load_json_dataset(settings.HOTELS_FILE)


@lru_cache(maxsize=1)
def _get_recommendation_engine() -> RecommendationEngine:
    """Build (and cache) the shared semantic recommendation engine."""
    return RecommendationEngine(hotels=_load_hotels_for_engine(), places=_load_places())


class PlaceSearchInput(BaseModel):
    """Input schema for :func:`search_places`."""

    city: str = Field(..., description="City to discover attractions in, e.g. 'Jaipur'")
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional travel-style category filter: one of 'Family', "
            "'Adventure', 'Historical', 'Cultural' or 'Relaxation'. Each "
            "place's raw type (fort, museum, beach, ...) is mapped to one "
            "or more of these categories."
        ),
    )
    preference_text: Optional[str] = Field(
        default=None,
        description=(
            "Optional free-text description of the traveller's interests "
            "(e.g. 'historic forts and cultural museums'). When provided, "
            "results are semantically re-ranked using a FAISS vector search."
        ),
    )
    top_k: int = Field(default=6, ge=1, le=40, description="Maximum number of results to return")


@log_tool_execution(TOOL_NAME)
def search_places(
    city: str,
    category: Optional[str] = None,
    preference_text: Optional[str] = None,
    top_k: int = 6,
) -> str:
    """Discover and rank tourist attractions in a given city.

    Args:
        city: City to discover attractions in.
        category: Optional travel-style category filter (Family, Adventure,
            Historical, Cultural, Relaxation).
        preference_text: Optional free-text preference used for semantic
            (FAISS) re-ranking.
        top_k: Maximum number of results to return.

    Returns:
        A JSON string encoding a :class:`models.response_models.ToolResponse`
        whose ``data`` field is a list of ranked place dictionaries, each
        annotated with a ``categories`` list.
    """
    try:
        city_name = validate_city(city, "city")
    except ValidationError as exc:
        return ToolResponse.fail(TOOL_NAME, error=str(exc)).to_json()

    if category is not None and category.strip().lower() not in VALID_CATEGORIES:
        return ToolResponse.fail(
            TOOL_NAME,
            error=(
                f"Invalid category '{category}'. Valid categories: "
                f"{', '.join(sorted(c.title() for c in VALID_CATEGORIES))}."
            ),
        ).to_json()

    places = _load_places()
    matches = [place for place in places if normalise_city(place["city"]) == city_name]

    if category:
        target_category = category.strip().title()
        matches = [
            place for place in matches if target_category in categorise_place(place["type"])
        ]

    if not matches:
        return ToolResponse.ok(
            TOOL_NAME,
            data=[],
            message=f"No attractions found in {city_name} matching the given filters.",
        ).to_json()

    ranked = rank_places(matches)

    if preference_text and preference_text.strip():
        engine = _get_recommendation_engine()
        semantic_results = engine.recommend_places(preference_text, top_k=len(ranked))
        semantic_scores = {p["place_id"]: p.get("semantic_score") for p in semantic_results}
        for place in ranked:
            place["semantic_score"] = semantic_scores.get(place["place_id"], 0.0)
        ranked = sorted(ranked, key=lambda p: p.get("semantic_score", 0.0), reverse=True)

    enriched = []
    for place in ranked[:top_k]:
        item = dict(place)
        item["categories"] = categorise_place(place["type"])
        enriched.append(item)

    return ToolResponse.ok(
        TOOL_NAME,
        data=enriched,
        message=(
            f"Found {len(matches)} attraction(s) in {city_name}"
            + (f" in category '{category}'" if category else "")
            + f"; returning top {len(enriched)}."
        ),
    ).to_json()


places_discovery_tool = StructuredTool.from_function(
    func=search_places,
    name=TOOL_NAME,
    description=(
        "Discover and rank tourist attractions/places of interest in a given "
        "city from the places dataset. Supports filtering by travel-style "
        "category ('Family', 'Adventure', 'Historical', 'Cultural', "
        "'Relaxation') and optional semantic re-ranking via free-text "
        "'preference_text'. Returns a JSON ToolResponse containing the "
        "ranked attractions."
    ),
    args_schema=PlaceSearchInput,
)
