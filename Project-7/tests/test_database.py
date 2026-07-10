"""Unit tests for :mod:`database.database`."""

from __future__ import annotations

import json


def test_insert_and_get_user_query(temp_database):
    query_id = temp_database.insert_user_query(
        session_id="session-1",
        source_city="Delhi",
        destination_city="Goa",
        start_date="2026-07-10",
        num_days=3,
        budget=50000,
        travel_style="Family",
        num_travellers=2,
        raw_query="Plan a trip from Delhi to Goa",
    )
    assert query_id >= 1

    queries = temp_database.get_user_queries(session_id="session-1")
    assert len(queries) == 1
    assert queries[0]["source_city"] == "Delhi"
    assert queries[0]["destination_city"] == "Goa"
    assert queries[0]["num_travellers"] == 2


def test_insert_and_get_itinerary(temp_database):
    query_id = temp_database.insert_user_query(
        session_id="session-2",
        source_city="Delhi",
        destination_city="Goa",
        start_date="2026-07-10",
        num_days=3,
        budget=50000,
        travel_style="Family",
        num_travellers=1,
    )

    itinerary_payload = {"trip_title": "Your 3-Day Trip to Goa", "num_days": 3}
    itinerary_id = temp_database.insert_itinerary(
        query_id=query_id,
        trip_title=itinerary_payload["trip_title"],
        itinerary=itinerary_payload,
        total_cost=25000.0,
        budget_category="Moderate",
    )
    assert itinerary_id >= 1

    itineraries = temp_database.get_itineraries(query_id=query_id)
    assert len(itineraries) == 1
    stored = itineraries[0]
    assert stored["trip_title"] == "Your 3-Day Trip to Goa"
    assert stored["budget_category"] == "Moderate"

    # itinerary_json should be parsed back into a dict by get_itineraries.
    assert isinstance(stored["itinerary_json"], dict)
    assert stored["itinerary_json"]["num_days"] == 3


def test_insert_flight_and_hotel_selection(temp_database):
    query_id = temp_database.insert_user_query(
        session_id="session-3",
        source_city="Bangalore",
        destination_city="Delhi",
        start_date="2026-08-01",
        num_days=2,
        budget=20000,
        travel_style="Adventure",
        num_travellers=1,
    )

    temp_database.insert_flight_selection(
        query_id=query_id,
        flight_id="FL0001",
        airline="IndiGo",
        source_city="Bangalore",
        destination_city="Delhi",
        price=5000,
        duration_minutes=150,
        selection_reason="Best value option.",
    )

    temp_database.insert_hotel_selection(
        query_id=query_id,
        hotel_id="HOT0001",
        hotel_name="City Stay",
        city="Delhi",
        stars=4,
        price_per_night=3000,
        selection_reason="Great rating for the price.",
    )

    # No dedicated getter for selections in the public API beyond the raw
    # tables, so verify indirectly via a direct query through _connect().
    with temp_database._connect() as conn:
        flight_rows = conn.execute(
            "SELECT * FROM flight_selections WHERE query_id = ?", (query_id,)
        ).fetchall()
        hotel_rows = conn.execute(
            "SELECT * FROM hotel_selections WHERE query_id = ?", (query_id,)
        ).fetchall()

    assert len(flight_rows) == 1
    assert flight_rows[0]["airline"] == "IndiGo"
    assert len(hotel_rows) == 1
    assert hotel_rows[0]["hotel_name"] == "City Stay"


def test_log_tool_call_and_get_search_logs(temp_database):
    log_id = temp_database.log_tool_call(
        session_id="session-4",
        tool_name="flight_search_tool",
        input_payload=json.dumps({"source": "Delhi", "destination": "Goa"}),
        output_status="success",
        duration_ms=12.5,
    )
    assert log_id >= 1

    logs = temp_database.get_search_logs(session_id="session-4")
    assert len(logs) == 1
    assert logs[0]["tool_name"] == "flight_search_tool"
    assert logs[0]["output_status"] == "success"
