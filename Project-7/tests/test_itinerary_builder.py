"""Unit tests for :mod:`services.itinerary_builder`."""

from __future__ import annotations

from datetime import date

from models.itinerary import Itinerary
from services.itinerary_builder import build_day_plans, build_itinerary, build_travel_tips

SAMPLE_PLACES = [
    {"place_id": "PLC0001", "name": "Old Fort", "city": "Goa", "type": "fort", "rating": 4.5},
    {"place_id": "PLC0002", "name": "City Museum", "city": "Goa", "type": "museum", "rating": 4.2},
]

SAMPLE_FLIGHT = {
    "flight_id": "FL0001",
    "airline": "IndiGo",
    "from": "Delhi",
    "to": "Goa",
    "departure_time": "2026-07-10T06:00:00",
    "arrival_time": "2026-07-10T08:30:00",
    "price": 5000,
    "duration_minutes": 150,
    "duration_label": "2h 30m",
    "airline_rating": 4.2,
    "value_score": 0.85,
}

SAMPLE_HOTEL = {
    "hotel_id": "HOT0001",
    "name": "Beach Resort",
    "city": "Goa",
    "stars": 4,
    "price_per_night": 3000,
    "amenities": ["wifi", "pool"],
    "value_score": 0.7,
    "semantic_score": 0.4,
}

SAMPLE_BUDGET = {
    "flight_cost": 10000.0,
    "hotel_cost": 9000.0,
    "food_cost": 2700.0,
    "local_transport_cost": 1500.0,
    "miscellaneous_cost": 2320.0,
    "total_cost": 25520.0,
    "daily_budget": 8506.67,
    "budget_category": "Moderate",
    "currency": "INR",
    "num_travellers": 2,
    "per_traveller_cost": 12760.0,
}

SAMPLE_REASONING = {
    "flight": "Chosen for best value.",
    "hotel": "Chosen for its rating and amenities.",
    "attractions": "Chosen for high ratings.",
    "itinerary_ordering": "Cycled across days and slots.",
}


def test_build_day_plans_with_places_cycles_through_slots():
    plans = build_day_plans(SAMPLE_PLACES, num_days=3, start_date=date(2026, 7, 10))
    assert len(plans) == 3
    places_found = sum(1 for plan in plans for activity in plan.activities if activity.place is not None)
    assert places_found == 2


def test_build_day_plans_without_places_returns_free_time():
    plans = build_day_plans([], num_days=2, start_date=date(2026, 7, 10))
    assert len(plans) == 2
    for plan in plans:
        for activity in plan.activities:
            assert activity.place is None
            assert activity.notes == "Free time / self-exploration"


def test_build_travel_tips_includes_style_specific_tip():
    tips = build_travel_tips("Goa", "Adventure", weather_forecast=[])
    assert any("adventure activities" in tip.lower() for tip in tips)


def test_build_travel_tips_includes_rain_tip_when_forecast_is_rainy():
    from models.itinerary import DailyWeather

    forecast = [
        DailyWeather(
            date="2026-07-10",
            condition="Heavy rain",
            temperature_max_c=28,
            temperature_min_c=24,
            precipitation_probability_pct=80,
        )
    ]
    tips = build_travel_tips("Goa", "Family", weather_forecast=forecast)
    assert any("umbrella" in tip.lower() or "raincoat" in tip.lower() for tip in tips)


def test_build_itinerary_assembles_full_model():
    itinerary = build_itinerary(
        source_city="Delhi",
        destination_city="Goa",
        start_date=date(2026, 7, 10),
        num_days=2,
        travel_style="Family",
        selected_flight=SAMPLE_FLIGHT,
        selected_hotel=SAMPLE_HOTEL,
        ranked_places=SAMPLE_PLACES,
        weather_forecast=[],
        budget_breakdown=SAMPLE_BUDGET,
        reasoning=SAMPLE_REASONING,
    )

    assert isinstance(itinerary, Itinerary)
    assert itinerary.source_city == "Delhi"
    assert itinerary.destination_city == "Goa"
    assert itinerary.end_date == "2026-07-11"
    assert itinerary.selected_flight is not None
    assert itinerary.selected_flight.flight_id == "FL0001"
    assert itinerary.selected_hotel is not None
    assert itinerary.budget is not None
    assert itinerary.budget.total_cost == 25520.0
    assert itinerary.reasoning is not None
    assert len(itinerary.day_plans) == 2
    assert itinerary.travel_tips


def test_build_itinerary_handles_missing_flight_and_hotel():
    itinerary = build_itinerary(
        source_city="Delhi",
        destination_city="Goa",
        start_date=date(2026, 7, 10),
        num_days=1,
        travel_style="Backpacker",
        selected_flight=None,
        selected_hotel=None,
        ranked_places=[],
        weather_forecast=[],
        budget_breakdown=SAMPLE_BUDGET,
        reasoning=SAMPLE_REASONING,
    )
    assert itinerary.selected_flight is None
    assert itinerary.selected_hotel is None


def test_itinerary_to_markdown_contains_key_sections():
    itinerary = build_itinerary(
        source_city="Delhi",
        destination_city="Goa",
        start_date=date(2026, 7, 10),
        num_days=2,
        travel_style="Family",
        selected_flight=SAMPLE_FLIGHT,
        selected_hotel=SAMPLE_HOTEL,
        ranked_places=SAMPLE_PLACES,
        weather_forecast=[],
        budget_breakdown=SAMPLE_BUDGET,
        reasoning=SAMPLE_REASONING,
    )
    markdown = itinerary.to_markdown()
    assert "# Executive Summary" in markdown
    assert "# Transportation" in markdown
    assert "# Attractions" in markdown
    assert "# Day-by-Day Itinerary" in markdown
    assert "# Budget Breakdown" in markdown
