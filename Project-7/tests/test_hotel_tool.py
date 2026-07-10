"""Unit tests for :mod:`tools.hotel_tool`."""

from __future__ import annotations

import json

from tools.hotel_tool import search_hotels


def test_search_hotels_known_city_returns_results():
    result = json.loads(search_hotels(city="Goa"))
    assert result["status"] == "success"
    assert result["data"], "expected at least one hotel in Goa"
    for hotel in result["data"]:
        assert hotel["city"] == "Goa"
        assert "value_score" in hotel


def test_search_hotels_invalid_city_returns_error():
    result = json.loads(search_hotels(city="Atlantis"))
    assert result["status"] == "error"


def test_search_hotels_price_filter():
    result = json.loads(search_hotels(city="Goa", max_price_per_night=2000))
    for hotel in result["data"]:
        assert hotel["price_per_night"] <= 2000


def test_search_hotels_min_stars_filter():
    result = json.loads(search_hotels(city="Goa", min_stars=4))
    for hotel in result["data"]:
        assert hotel["stars"] >= 4


def test_search_hotels_filters_with_no_match_returns_empty():
    result = json.loads(search_hotels(city="Goa", max_price_per_night=1))
    assert result["status"] == "success"
    assert result["data"] == []
    assert "No hotels found" in result["message"]


def test_search_hotels_sort_by_price_ascending():
    result = json.loads(search_hotels(city="Goa", sort_by="price", top_k=20))
    prices = [h["price_per_night"] for h in result["data"]]
    assert prices == sorted(prices)


def test_search_hotels_sort_by_rating_descending():
    result = json.loads(search_hotels(city="Goa", sort_by="rating", top_k=20))
    stars = [h["stars"] for h in result["data"]]
    assert stars == sorted(stars, reverse=True)


def test_search_hotels_semantic_rerank_adds_semantic_score():
    result = json.loads(
        search_hotels(city="Goa", preference_text="hotel with pool and spa", top_k=5)
    )
    assert result["status"] == "success"
    for hotel in result["data"]:
        assert "semantic_score" in hotel


def test_search_hotels_top_k_limits_results():
    result = json.loads(search_hotels(city="Goa", top_k=2))
    assert len(result["data"]) <= 2
