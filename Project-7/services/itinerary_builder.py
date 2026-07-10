"""
Itinerary builder service.

Combines the outputs of the flight, hotel, places, weather and budget tools
into a single structured :class:`models.itinerary.Itinerary` object -- the
day-wise plan (Morning/Afternoon/Evening), weather-aware notes, budget
breakdown and transparent reasoning trace described in the project brief.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import timedelta
from typing import Any, Dict, List, Optional

from configs import settings
from models.itinerary import (
    AttractionOption,
    BudgetBreakdown,
    DailyWeather,
    DayPlan,
    DaySlotActivity,
    FlightOption,
    HotelOption,
    Itinerary,
    ReasoningTrace,
)
from utils.helpers import categorise_place
from utils.logger import get_logger

logger = get_logger("itinerary_builder")


def _to_flight_option(flight: Optional[Dict[str, Any]]) -> Optional[FlightOption]:
    if not flight:
        return None
    return FlightOption(
        flight_id=flight["flight_id"],
        airline=flight["airline"],
        **{"from": flight["from"], "to": flight["to"]},
        departure_time=flight["departure_time"],
        arrival_time=flight["arrival_time"],
        price=flight["price"],
        duration_minutes=flight.get("duration_minutes", 0),
        duration_label=flight.get("duration_label", ""),
        airline_rating=flight.get("airline_rating", settings.DEFAULT_AIRLINE_RATING),
        value_score=flight.get("value_score"),
    )


def _to_hotel_option(hotel: Optional[Dict[str, Any]]) -> Optional[HotelOption]:
    if not hotel:
        return None
    return HotelOption(
        hotel_id=hotel["hotel_id"],
        name=hotel["name"],
        city=hotel["city"],
        stars=hotel["stars"],
        price_per_night=hotel["price_per_night"],
        amenities=hotel.get("amenities", []),
        value_score=hotel.get("value_score"),
        semantic_score=hotel.get("semantic_score"),
    )


def _to_attraction_option(place: Dict[str, Any]) -> AttractionOption:
    return AttractionOption(
        place_id=place["place_id"],
        name=place["name"],
        city=place["city"],
        type=place["type"],
        categories=categorise_place(place["type"]),
        rating=place["rating"],
        semantic_score=place.get("semantic_score"),
    )


def build_day_plans(
    places: List[Dict[str, Any]],
    num_days: int,
    start_date: date_type,
    weather_by_date: Optional[Dict[str, DailyWeather]] = None,
) -> List[DayPlan]:
    """Distribute attractions across days and Morning/Afternoon/Evening slots.

    Places are assumed to be pre-ranked best-first (e.g. by rating). The
    function walks the ranked list and assigns up to
    ``len(settings.SLOTS_PER_DAY)`` attractions per day, cycling through
    available places if there are fewer unique places than slots so every
    day still has a plan.

    Args:
        places: Ranked list of place dicts (best first).
        num_days: Number of days in the trip.
        start_date: First day of the trip.
        weather_by_date: Optional mapping of ``YYYY-MM-DD`` -> DailyWeather,
            attached to each day for quick reference.

    Returns:
        A list of :class:`DayPlan` objects, one per day.
    """
    weather_by_date = weather_by_date or {}
    slots = settings.SLOTS_PER_DAY
    day_plans: List[DayPlan] = []

    if not places:
        for day_index in range(num_days):
            current_date = (start_date + timedelta(days=day_index)).isoformat()
            activities = [
                DaySlotActivity(slot=slot, place=None, notes="Free time / self-exploration")
                for slot in slots
            ]
            day_plans.append(
                DayPlan(
                    day_number=day_index + 1,
                    date=current_date,
                    weather=weather_by_date.get(current_date),
                    activities=activities,
                )
            )
        return day_plans

    total_slots = num_days * len(slots)
    cursor = 0
    for day_index in range(num_days):
        current_date = (start_date + timedelta(days=day_index)).isoformat()
        activities: List[DaySlotActivity] = []
        for slot in slots:
            if cursor < len(places):
                place = places[cursor]
                activities.append(DaySlotActivity(slot=slot, place=_to_attraction_option(place)))
                cursor += 1
            else:
                activities.append(DaySlotActivity(slot=slot, place=None, notes="Free time / self-exploration"))
        day_plans.append(
            DayPlan(
                day_number=day_index + 1,
                date=current_date,
                weather=weather_by_date.get(current_date),
                activities=activities,
            )
        )

    logger.debug(
        "Built {} day plans covering {} attraction slots from {} candidate places",
        num_days,
        total_slots,
        len(places),
    )
    return day_plans


def build_travel_tips(
    destination_city: str,
    travel_style: str,
    weather_forecast: List[DailyWeather],
) -> List[str]:
    """Generate a short list of practical, weather- and style-aware travel tips."""
    tips: List[str] = [
        f"Carry a valid government-issued photo ID; it is commonly required for "
        f"hotel check-in and domestic flights within India.",
        f"Keep a digital and printed copy of your {destination_city} hotel booking "
        f"and flight ticket.",
    ]

    rainy = any(
        "rain" in (day.condition or "").lower() or (day.precipitation_probability_pct or 0) >= 50
        for day in weather_forecast
    )
    hot = any((day.temperature_max_c or 0) >= 33 for day in weather_forecast)
    cold = any((day.temperature_min_c or 99) <= 15 for day in weather_forecast)

    if rainy:
        tips.append("Pack a compact umbrella or raincoat -- rain is likely on at least one day.")
    if hot:
        tips.append("Expect high daytime temperatures; carry water, sunscreen, and light cotton clothing.")
    if cold:
        tips.append("Evenings/mornings may be cool; pack a light jacket or sweater.")

    style_tips = {
        "Family": "Plan indoor/AC breaks during midday heat for younger travellers.",
        "Adventure": "Book adventure activities (water sports, treks) in advance and check weather-based cancellations.",
        "Luxury": "Pre-book spa slots and fine-dining reservations at your hotel.",
        "Backpacker": "Use local trains/buses and shared cabs to keep local transport costs low.",
    }
    if travel_style in style_tips:
        tips.append(style_tips[travel_style])

    tips.append("Keep some local currency cash on hand for markets and small vendors that may not accept cards.")
    return tips


def build_itinerary(
    *,
    source_city: str,
    destination_city: str,
    start_date: date_type,
    num_days: int,
    travel_style: str,
    selected_flight: Optional[Dict[str, Any]],
    selected_hotel: Optional[Dict[str, Any]],
    ranked_places: List[Dict[str, Any]],
    weather_forecast: List[Dict[str, Any]],
    budget_breakdown: Dict[str, Any],
    reasoning: Dict[str, str],
) -> Itinerary:
    """Assemble the final :class:`Itinerary` from all tool outputs.

    Args:
        source_city: Departure city.
        destination_city: Destination city.
        start_date: Trip start date.
        num_days: Trip length in days.
        travel_style: Travel style used for filtering/tips.
        selected_flight: The flight record chosen by the agent (enriched).
        selected_hotel: The hotel record chosen by the agent (enriched).
        ranked_places: Ranked list of attraction dicts to distribute across days.
        weather_forecast: List of daily weather dicts from the Weather Tool.
        budget_breakdown: Output of the Budget Estimation Tool.
        reasoning: Dict with keys ``flight``, ``hotel``, ``attractions``,
            ``itinerary_ordering`` containing human-readable explanations.

    Returns:
        A fully populated :class:`Itinerary`.
    """
    weather_models = [
        DailyWeather(
            date=day["date"],
            condition=day["condition"],
            temperature_max_c=day.get("temperature_max_c"),
            temperature_min_c=day.get("temperature_min_c"),
            precipitation_probability_pct=day.get("precipitation_probability_pct"),
        )
        for day in weather_forecast
    ]
    weather_by_date = {day.date: day for day in weather_models}

    # Validation: Destination consistency
    if selected_flight and selected_flight.get("to") and selected_flight["to"].lower() != destination_city.lower():
        logger.warning("Discarding invalid flight to {} (expected {})", selected_flight["to"], destination_city)
        selected_flight = None

    if selected_hotel and selected_hotel.get("city") and selected_hotel["city"].lower() != destination_city.lower():
        logger.warning("Discarding invalid hotel in {} (expected {})", selected_hotel["city"], destination_city)
        selected_hotel = None

    day_plans = build_day_plans(ranked_places, num_days, start_date, weather_by_date)
    travel_tips = build_travel_tips(destination_city, travel_style, weather_models)

    end_date = start_date + timedelta(days=num_days - 1)

    itinerary = Itinerary(
        trip_title=f"Your {num_days}-Day Trip to {destination_city}",
        source_city=source_city,
        destination_city=destination_city,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        num_days=num_days,
        travel_style=travel_style,
        selected_flight=_to_flight_option(selected_flight),
        selected_hotel=_to_hotel_option(selected_hotel),
        weather_forecast=weather_models,
        day_plans=day_plans,
        budget=BudgetBreakdown(**budget_breakdown),
        reasoning=ReasoningTrace(
            flight_reasoning=reasoning.get("flight", "Not available."),
            hotel_reasoning=reasoning.get("hotel", "Not available."),
            attractions_reasoning=reasoning.get("attractions", "Not available."),
            itinerary_ordering_reasoning=reasoning.get("itinerary_ordering", "Not available."),
        ),
        travel_tips=travel_tips,
    )

    logger.info(
        "Built itinerary for {} -> {} ({} days, style={})",
        source_city,
        destination_city,
        num_days,
        travel_style,
    )
    return itinerary
