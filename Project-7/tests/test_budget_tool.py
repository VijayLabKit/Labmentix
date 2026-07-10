"""Unit tests for :mod:`tools.budget_tool`."""

from __future__ import annotations

import json

from configs import settings
from tools.budget_tool import estimate_budget


def test_estimate_budget_basic_breakdown_keys():
    result = json.loads(
        estimate_budget(flight_price=5000, hotel_price_per_night=2000, num_days=3, num_travellers=2)
    )
    assert result["status"] == "success"
    data = result["data"]
    expected_keys = {
        "flight_cost",
        "hotel_cost",
        "food_cost",
        "local_transport_cost",
        "miscellaneous_cost",
        "total_cost",
        "daily_budget",
        "budget_category",
        "currency",
        "num_travellers",
        "per_traveller_cost",
    }
    assert expected_keys.issubset(data.keys())
    assert data["currency"] == "INR"
    assert data["num_travellers"] == 2


def test_estimate_budget_arithmetic_is_consistent():
    flight_price, hotel_price, num_days, num_travellers = 5000, 2000, 3, 2
    result = json.loads(
        estimate_budget(
            flight_price=flight_price,
            hotel_price_per_night=hotel_price,
            num_days=num_days,
            num_travellers=num_travellers,
        )
    )
    data = result["data"]

    expected_flight_cost = flight_price * num_travellers
    expected_hotel_cost = hotel_price * num_days
    expected_food_cost = settings.DAILY_FOOD_COST_INR * num_days * num_travellers
    expected_transport_cost = settings.DAILY_LOCAL_TRANSPORT_COST_INR * num_days * num_travellers

    assert data["flight_cost"] == expected_flight_cost
    assert data["hotel_cost"] == expected_hotel_cost
    assert data["food_cost"] == expected_food_cost
    assert data["local_transport_cost"] == expected_transport_cost

    subtotal = (
        expected_flight_cost + expected_hotel_cost + expected_food_cost + expected_transport_cost
    )
    expected_misc = round(subtotal * settings.MISCELLANEOUS_RATE, 2)
    expected_total = round(subtotal + expected_misc, 2)

    assert data["miscellaneous_cost"] == expected_misc
    assert data["total_cost"] == expected_total
    assert data["daily_budget"] == round(expected_total / num_days, 2)
    assert data["per_traveller_cost"] == round(expected_total / num_travellers, 2)


def test_estimate_budget_category_thresholds():
    # A very cheap, single-traveller, single-day trip should fall in the
    # 'Budget' category.
    cheap = json.loads(
        estimate_budget(flight_price=500, hotel_price_per_night=500, num_days=1, num_travellers=1)
    )
    assert cheap["data"]["budget_category"] == "Budget"

    # A very expensive trip should fall in the 'Luxury' category.
    expensive = json.loads(
        estimate_budget(
            flight_price=50000, hotel_price_per_night=20000, num_days=5, num_travellers=1
        )
    )
    assert expensive["data"]["budget_category"] == "Luxury"


def test_estimate_budget_rejects_invalid_num_days():
    result = json.loads(
        estimate_budget(flight_price=1000, hotel_price_per_night=1000, num_days=0, num_travellers=1)
    )
    assert result["status"] == "error"


def test_estimate_budget_rejects_invalid_num_travellers():
    result = json.loads(
        estimate_budget(flight_price=1000, hotel_price_per_night=1000, num_days=1, num_travellers=0)
    )
    assert result["status"] == "error"


def test_estimate_budget_matches_budget_breakdown_model():
    from models.itinerary import BudgetBreakdown

    result = json.loads(
        estimate_budget(flight_price=4000, hotel_price_per_night=3000, num_days=2, num_travellers=2)
    )
    # Should construct without raising - i.e. keys/types are compatible.
    breakdown = BudgetBreakdown(**result["data"])
    assert breakdown.total_cost > 0
    assert breakdown.budget_category in {"Budget", "Moderate", "Comfort", "Luxury"}
