"""
Ranking engine: pure scoring/sorting functions for flights, hotels and
places of interest.

Every function in this module is a pure transformation over plain Python
dictionaries (the records as loaded from ``data/*.json``) and returns new
dictionaries with additional derived fields (``duration_minutes``,
``value_score``, etc.). Keeping these functions pure and dependency-light
(no LangChain) makes them straightforward to unit test and reuse from both
the LangChain tools and the Streamlit UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from configs import settings
from utils.helpers import (
    flight_duration_minutes,
    format_duration,
    get_airline_rating,
    min_max_normalise,
)

FlightCriteria = Literal["cheapest", "fastest", "best_value"]
HotelCriteria = Literal["price", "rating", "best_value"]


# ---------------------------------------------------------------------------
# Flights
# ---------------------------------------------------------------------------


def enrich_flight(flight: Dict[str, Any]) -> Dict[str, Any]:
    """Add derived fields (duration, airline rating) to a flight record."""
    enriched = dict(flight)
    duration = flight_duration_minutes(flight["departure_time"], flight["arrival_time"])
    enriched["duration_minutes"] = duration
    enriched["duration_label"] = format_duration(duration)
    enriched["airline_rating"] = get_airline_rating(flight["airline"])
    return enriched


def _flight_value_scores(flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach a normalised 'value_score' (higher is better) to each flight.

    The score combines price, duration and airline rating using the weights
    defined in :data:`configs.settings.FLIGHT_RANKING_WEIGHTS`. Price and
    duration are inverted (lower is better -> higher score contribution).
    """
    if not flights:
        return []

    prices = [f["price"] for f in flights]
    durations = [f["duration_minutes"] for f in flights]
    ratings = [f["airline_rating"] for f in flights]

    price_min, price_max = min(prices), max(prices)
    duration_min, duration_max = min(durations), max(durations)
    rating_min, rating_max = min(ratings), max(ratings)

    weights = settings.FLIGHT_RANKING_WEIGHTS
    scored = []
    for flight in flights:
        price_norm = 1 - min_max_normalise(flight["price"], price_min, price_max)
        duration_norm = 1 - min_max_normalise(flight["duration_minutes"], duration_min, duration_max)
        rating_norm = min_max_normalise(flight["airline_rating"], rating_min, rating_max)

        value_score = (
            weights["price"] * price_norm
            + weights["duration"] * duration_norm
            + weights["airline_rating"] * rating_norm
        )
        enriched = dict(flight)
        enriched["value_score"] = round(value_score, 4)
        scored.append(enriched)
    return scored


def rank_flights(
    flights: List[Dict[str, Any]], criteria: FlightCriteria = "best_value"
) -> List[Dict[str, Any]]:
    """Rank a list of flight records according to ``criteria``.

    Args:
        flights: Flight records, typically already filtered by
            source/destination/date.
        criteria: One of ``"cheapest"``, ``"fastest"`` or ``"best_value"``.

    Returns:
        A new list of flight dicts (each enriched with ``duration_minutes``,
        ``duration_label`` and ``airline_rating``), sorted best-first
        according to ``criteria``. ``best_value`` additionally includes a
        ``value_score`` field.
    """
    enriched = [enrich_flight(f) for f in flights]

    if criteria == "cheapest":
        return sorted(enriched, key=lambda f: f["price"])
    if criteria == "fastest":
        return sorted(enriched, key=lambda f: f["duration_minutes"])

    scored = _flight_value_scores(enriched)
    return sorted(scored, key=lambda f: f["value_score"], reverse=True)


# ---------------------------------------------------------------------------
# Hotels
# ---------------------------------------------------------------------------


def _hotel_value_scores(hotels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach a normalised 'value_score' (higher is better) to each hotel.

    Combines (inverted) price and star rating using
    :data:`configs.settings.HOTEL_RANKING_WEIGHTS`.
    """
    if not hotels:
        return []

    prices = [h["price_per_night"] for h in hotels]
    stars = [h["stars"] for h in hotels]

    price_min, price_max = min(prices), max(prices)
    star_min, star_max = min(stars), max(stars)

    weights = settings.HOTEL_RANKING_WEIGHTS
    scored = []
    for hotel in hotels:
        price_norm = 1 - min_max_normalise(hotel["price_per_night"], price_min, price_max)
        star_norm = min_max_normalise(hotel["stars"], star_min, star_max)
        value_score = weights["price"] * price_norm + weights["rating"] * star_norm
        enriched = dict(hotel)
        enriched["value_score"] = round(value_score, 4)
        scored.append(enriched)
    return scored


def rank_hotels(
    hotels: List[Dict[str, Any]], criteria: HotelCriteria = "best_value"
) -> List[Dict[str, Any]]:
    """Rank a list of hotel records according to ``criteria``.

    Args:
        hotels: Hotel records, typically already filtered by city/budget.
        criteria: One of ``"price"`` (cheapest first), ``"rating"`` (highest
            star rating first), or ``"best_value"``.

    Returns:
        A new, sorted list of hotel dicts. ``best_value`` additionally
        includes a ``value_score`` field.
    """
    if criteria == "price":
        return sorted(hotels, key=lambda h: h["price_per_night"])
    if criteria == "rating":
        return sorted(hotels, key=lambda h: h["stars"], reverse=True)

    scored = _hotel_value_scores(hotels)
    return sorted(scored, key=lambda h: h["value_score"], reverse=True)


# ---------------------------------------------------------------------------
# Places / attractions
# ---------------------------------------------------------------------------


def rank_places(places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank places of interest by rating (descending).

    Args:
        places: Place records, typically already filtered by city/category.

    Returns:
        A new list sorted by ``rating`` descending.
    """
    return sorted(places, key=lambda p: p.get("rating", 0), reverse=True)
