"""Unit tests for :mod:`services.ranking_engine`."""

from __future__ import annotations

from services.ranking_engine import enrich_flight, rank_flights, rank_hotels, rank_places


SAMPLE_FLIGHTS = [
    {
        "flight_id": "FL0001",
        "airline": "IndiGo",
        "from": "Delhi",
        "to": "Goa",
        "departure_time": "2026-07-10T06:00:00",
        "arrival_time": "2026-07-10T08:30:00",
        "price": 5000,
    },
    {
        "flight_id": "FL0002",
        "airline": "Go First",
        "from": "Delhi",
        "to": "Goa",
        "departure_time": "2026-07-10T06:00:00",
        "arrival_time": "2026-07-10T09:30:00",
        "price": 3000,
    },
    {
        "flight_id": "FL0003",
        "airline": "Vistara",
        "from": "Delhi",
        "to": "Goa",
        "departure_time": "2026-07-10T06:00:00",
        "arrival_time": "2026-07-10T08:00:00",
        "price": 7000,
    },
]

SAMPLE_HOTELS = [
    {"hotel_id": "HOT0001", "name": "Cheap Inn", "city": "Goa", "stars": 2, "price_per_night": 1000, "amenities": ["wifi"]},
    {"hotel_id": "HOT0002", "name": "Mid Stay", "city": "Goa", "stars": 4, "price_per_night": 3000, "amenities": ["wifi", "pool"]},
    {"hotel_id": "HOT0003", "name": "Lux Resort", "city": "Goa", "stars": 5, "price_per_night": 8000, "amenities": ["wifi", "pool", "spa"]},
]

SAMPLE_PLACES = [
    {"place_id": "PLC0001", "name": "A", "city": "Goa", "type": "beach", "rating": 4.0},
    {"place_id": "PLC0002", "name": "B", "city": "Goa", "type": "fort", "rating": 4.8},
    {"place_id": "PLC0003", "name": "C", "city": "Goa", "type": "market", "rating": 3.5},
]


def test_enrich_flight_adds_duration_and_rating():
    enriched = enrich_flight(SAMPLE_FLIGHTS[0])
    assert enriched["duration_minutes"] == 150
    assert enriched["duration_label"] == "2h 30m"
    assert enriched["airline_rating"] == 4.2


def test_rank_flights_cheapest():
    ranked = rank_flights(SAMPLE_FLIGHTS, criteria="cheapest")
    assert [f["flight_id"] for f in ranked] == ["FL0002", "FL0001", "FL0003"]
    # cheapest ordering does not require a value_score field
    assert "value_score" not in ranked[0]


def test_rank_flights_fastest():
    ranked = rank_flights(SAMPLE_FLIGHTS, criteria="fastest")
    assert ranked[0]["flight_id"] == "FL0003"  # 2h, the shortest
    assert ranked[0]["duration_minutes"] == 120


def test_rank_flights_best_value_adds_score_and_sorts_descending():
    ranked = rank_flights(SAMPLE_FLIGHTS, criteria="best_value")
    assert all("value_score" in f for f in ranked)
    scores = [f["value_score"] for f in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_flights_empty_input():
    assert rank_flights([], criteria="best_value") == []


def test_rank_hotels_price():
    ranked = rank_hotels(SAMPLE_HOTELS, criteria="price")
    assert [h["hotel_id"] for h in ranked] == ["HOT0001", "HOT0002", "HOT0003"]


def test_rank_hotels_rating():
    ranked = rank_hotels(SAMPLE_HOTELS, criteria="rating")
    assert ranked[0]["stars"] == 5


def test_rank_hotels_best_value_adds_score():
    ranked = rank_hotels(SAMPLE_HOTELS, criteria="best_value")
    assert all("value_score" in h for h in ranked)
    scores = [h["value_score"] for h in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_places_sorts_by_rating_descending():
    ranked = rank_places(SAMPLE_PLACES)
    assert [p["place_id"] for p in ranked] == ["PLC0002", "PLC0001", "PLC0003"]


def test_rank_flights_with_real_dataset(flights_data):
    ranked = rank_flights(flights_data, criteria="best_value")
    assert len(ranked) == len(flights_data)
    assert all("duration_minutes" in f and "value_score" in f for f in ranked)


def test_rank_hotels_with_real_dataset(hotels_data):
    ranked = rank_hotels(hotels_data, criteria="best_value")
    assert len(ranked) == len(hotels_data)
    assert all("value_score" in h for h in ranked)
