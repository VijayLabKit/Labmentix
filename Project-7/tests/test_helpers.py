"""Unit tests for :mod:`utils.helpers`."""

from __future__ import annotations

from datetime import datetime

import pytest

from utils import helpers


def test_load_json_dataset_returns_list(flights_data, hotels_data, places_data):
    assert isinstance(flights_data, list) and flights_data
    assert isinstance(hotels_data, list) and hotels_data
    assert isinstance(places_data, list) and places_data


def test_load_json_dataset_missing_file_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    assert helpers.load_json_dataset(missing) == []


def test_load_json_dataset_rejects_non_array(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError):
        helpers.load_json_dataset(bad_file)


def test_flight_duration_minutes_same_day():
    minutes = helpers.flight_duration_minutes(
        "2026-07-10T08:00:00", "2026-07-10T10:30:00"
    )
    assert minutes == 150


def test_flight_duration_minutes_overnight():
    minutes = helpers.flight_duration_minutes(
        "2026-07-10T23:00:00", "2026-07-11T01:00:00"
    )
    assert minutes == 120


def test_format_duration():
    assert helpers.format_duration(150) == "2h 30m"
    assert helpers.format_duration(120) == "2h"
    assert helpers.format_duration(45) == "45m"
    assert helpers.format_duration(0) == "0m"


def test_normalise_city():
    assert helpers.normalise_city("  delhi ") == "Delhi"
    assert helpers.normalise_city("NEW DELHI") == "New Delhi"


def test_get_city_coordinates_known_and_unknown():
    assert helpers.get_city_coordinates("goa") == (15.2993, 74.1240)
    assert helpers.get_city_coordinates("Atlantis") is None


def test_get_airline_rating_known_and_default():
    assert helpers.get_airline_rating("IndiGo") == 4.2
    assert helpers.get_airline_rating("Unknown Airline") == 3.5


@pytest.mark.parametrize(
    "place_type,expected_first",
    [
        ("beach", "Adventure"),
        ("fort", "Historical"),
        ("museum", "Cultural"),
        ("unknown_type", "Cultural"),
    ],
)
def test_categorise_place(place_type, expected_first):
    categories = helpers.categorise_place(place_type)
    assert expected_first in categories


def test_min_max_normalise():
    assert helpers.min_max_normalise(5, 0, 10) == 0.5
    assert helpers.min_max_normalise(0, 0, 10) == 0.0
    assert helpers.min_max_normalise(10, 0, 10) == 1.0
    # Equal bounds should not raise and should return the neutral midpoint.
    assert helpers.min_max_normalise(5, 5, 5) == 0.5


def test_daterange():
    start = datetime(2026, 7, 1)
    days = helpers.daterange(start, 3)
    assert [d.day for d in days] == [1, 2, 3]


def test_chunked():
    chunks = list(helpers.chunked(range(7), 3))
    assert chunks == [[0, 1, 2], [3, 4, 5], [6]]


def test_collapse_whitespace():
    assert helpers.collapse_whitespace("  a   b\nc\t d ") == "a b c d"


def test_safe_round():
    assert helpers.safe_round(1.005, 2) in (1.0, 1.01)  # float-precision tolerant
    assert helpers.safe_round(None) == 0.0
    assert helpers.safe_round("not-a-number") == 0.0
    assert helpers.safe_round(3.14159, 2) == 3.14
