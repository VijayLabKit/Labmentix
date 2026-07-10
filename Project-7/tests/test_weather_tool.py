"""Unit tests for :mod:`tools.weather_tool`.

All tests mock ``requests.get`` so the suite runs fully offline and fast,
regardless of whether the sandbox can reach the real Open-Meteo API.
"""

from __future__ import annotations

import json

import pytest
import requests

from tools import weather_tool
from tools.weather_tool import get_weather_forecast, weather_code_to_condition


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


def _open_meteo_payload(dates):
    n = len(dates)
    return {
        "daily": {
            "time": dates,
            "temperature_2m_max": [30.0 + i for i in range(n)],
            "temperature_2m_min": [20.0 + i for i in range(n)],
            "precipitation_probability_max": [10 * i for i in range(n)],
            "weathercode": [0, 61, 95][:n] + [0] * max(0, n - 3),
        }
    }


def test_weather_code_to_condition_known_and_unknown():
    assert weather_code_to_condition(0) == "Clear sky"
    assert weather_code_to_condition(61) == "Slight rain"
    assert weather_code_to_condition(None) == "Unknown"
    assert weather_code_to_condition(12345) == "Unknown"


def test_get_weather_forecast_success(monkeypatch):
    dates = ["2026-07-10", "2026-07-11", "2026-07-12"]

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(_open_meteo_payload(dates))

    monkeypatch.setattr(weather_tool.requests, "get", fake_get)

    result = json.loads(get_weather_forecast(city="Goa", start_date="2026-07-10", num_days=3))
    assert result["status"] == "success"
    assert len(result["data"]) == 3
    assert result["data"][0]["condition"] == "Clear sky"
    assert result["data"][1]["condition"] == "Slight rain"
    assert result["data"][0]["source"] == "open-meteo"
    assert result["data"][0]["temperature_max_c"] == 30.0


def test_get_weather_forecast_invalid_city():
    result = json.loads(get_weather_forecast(city="Atlantis", start_date="2026-07-10"))
    assert result["status"] == "error"


def test_get_weather_forecast_invalid_date():
    result = json.loads(get_weather_forecast(city="Goa", start_date="10-07-2026"))
    assert result["status"] == "error"
    assert "Invalid start_date" in result["error"]


def test_get_weather_forecast_network_failure_returns_error_after_retries(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, params=None, timeout=None):
        calls["count"] += 1
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(weather_tool.requests, "get", fake_get)
    monkeypatch.setattr(weather_tool.time, "sleep", lambda *_args, **_kwargs: None)

    result = json.loads(get_weather_forecast(city="Goa", start_date="2026-07-10", num_days=2))
    assert result["status"] == "error"
    assert "Failed to fetch weather data" in result["error"]
    assert calls["count"] == weather_tool.settings.WEATHER_MAX_RETRIES


def test_get_weather_forecast_empty_response_returns_ok_with_message(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse({"daily": {"time": []}})

    monkeypatch.setattr(weather_tool.requests, "get", fake_get)

    result = json.loads(get_weather_forecast(city="Goa", start_date="2026-07-10", num_days=2))
    assert result["status"] == "success"
    assert result["data"] == []
    assert "no forecast data" in result["message"]
