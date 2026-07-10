"""Unit tests for :mod:`tools.flight_tool`."""

from __future__ import annotations

import json

from tools.flight_tool import search_flights


def test_search_flights_known_route_returns_results():
    result = json.loads(search_flights(source="Bangalore", destination="Delhi"))
    assert result["status"] == "success"
    assert result["data"], "expected at least one flight on a known route"
    for flight in result["data"]:
        assert flight["from"] == "Bangalore"
        assert flight["to"] == "Delhi"
        assert "duration_minutes" in flight
        assert "duration_label" in flight
        assert "airline_rating" in flight


def test_search_flights_unknown_route_returns_empty_with_message():
    result = json.loads(search_flights(source="Delhi", destination="Mumbai"))
    assert result["status"] == "success"
    assert result["data"] == []
    assert "No flights found" in result["message"]


def test_search_flights_invalid_city_returns_error():
    result = json.loads(search_flights(source="Atlantis", destination="Goa"))
    assert result["status"] == "error"
    assert "not supported" in result["error"]


def test_search_flights_cheapest_criteria_sorted_ascending():
    result = json.loads(
        search_flights(source="Bangalore", destination="Delhi", criteria="cheapest", top_k=10)
    )
    prices = [f["price"] for f in result["data"]]
    assert prices == sorted(prices)


def test_search_flights_fastest_criteria_sorted_ascending_duration():
    result = json.loads(
        search_flights(source="Bangalore", destination="Delhi", criteria="fastest", top_k=10)
    )
    durations = [f["duration_minutes"] for f in result["data"]]
    assert durations == sorted(durations)


def test_search_flights_top_k_limits_results():
    result = json.loads(
        search_flights(source="Bangalore", destination="Delhi", criteria="best_value", top_k=1)
    )
    assert len(result["data"]) <= 1


def test_search_flights_invalid_date_format_returns_error():
    result = json.loads(
        search_flights(source="Bangalore", destination="Delhi", travel_date="10-07-2026")
    )
    assert result["status"] == "error"
    assert "Invalid travel_date" in result["error"]
