"""Unit tests for :mod:`services.recommendation_engine`."""

from __future__ import annotations

from services.recommendation_engine import RecommendationEngine


def test_recommend_hotels_returns_scored_results(hotels_data, places_data):
    engine = RecommendationEngine(hotels_data, places_data)
    results = engine.recommend_hotels("luxury hotel with spa and pool", top_k=5)

    assert results, "expected at least one hotel recommendation"
    assert len(results) <= 5
    for hotel in results:
        assert "semantic_score" in hotel
        assert -1.0001 <= hotel["semantic_score"] <= 1.0001

    scores = [h["semantic_score"] for h in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_places_returns_scored_results(hotels_data, places_data):
    engine = RecommendationEngine(hotels_data, places_data)
    results = engine.recommend_places("historic forts and museums", top_k=5)

    assert results, "expected at least one place recommendation"
    for place in results:
        assert "semantic_score" in place

    scores = [p["semantic_score"] for p in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_with_empty_query_returns_empty(hotels_data, places_data):
    engine = RecommendationEngine(hotels_data, places_data)
    assert engine.recommend_hotels("", top_k=5) == []
    assert engine.recommend_places("   ", top_k=5) == []


def test_recommend_with_empty_records_returns_empty():
    engine = RecommendationEngine([], [])
    assert engine.recommend_hotels("anything", top_k=5) == []
    assert engine.recommend_places("anything", top_k=5) == []


def test_recommend_top_k_is_respected(hotels_data, places_data):
    engine = RecommendationEngine(hotels_data, places_data)
    results = engine.recommend_hotels("hotel", top_k=2)
    assert len(results) <= 2
