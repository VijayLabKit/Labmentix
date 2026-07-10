"""Unit tests for :mod:`tools.place_tool`."""

from __future__ import annotations

import json

from tools.place_tool import search_places


def test_search_places_known_city_returns_results():
    result = json.loads(search_places(city="Goa"))
    assert result["status"] == "success"
    assert result["data"], "expected at least one attraction in Goa"
    for place in result["data"]:
        assert place["city"] == "Goa"
        assert "categories" in place


def test_search_places_invalid_city_returns_error():
    result = json.loads(search_places(city="Atlantis"))
    assert result["status"] == "error"


def test_search_places_invalid_category_returns_error():
    result = json.loads(search_places(city="Goa", category="Nonsense"))
    assert result["status"] == "error"
    assert "Invalid category" in result["error"]


def test_search_places_category_filter_returns_matching_categories():
    result = json.loads(search_places(city="Goa", category="Historical", top_k=40))
    if result["data"]:
        for place in result["data"]:
            assert "Historical" in place["categories"]
    else:
        assert "No attractions found" in result["message"]


def test_search_places_sorted_by_rating_descending():
    result = json.loads(search_places(city="Goa", top_k=40))
    ratings = [p["rating"] for p in result["data"]]
    assert ratings == sorted(ratings, reverse=True)


def test_search_places_semantic_rerank_adds_semantic_score():
    result = json.loads(
        search_places(city="Goa", preference_text="historic forts and museums", top_k=5)
    )
    assert result["status"] == "success"
    for place in result["data"]:
        assert "semantic_score" in place


def test_search_places_top_k_limits_results():
    result = json.loads(search_places(city="Goa", top_k=3))
    assert len(result["data"]) <= 3
