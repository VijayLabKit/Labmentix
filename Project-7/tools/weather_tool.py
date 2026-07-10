"""
Weather Lookup Tool.

Calls the free Open-Meteo forecast API (no API key required) to retrieve a
daily weather forecast for a destination city, with retry/backoff logic for
transient network errors.

API reference: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import time
from datetime import date as date_type
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from configs import settings
from models.response_models import ToolResponse
from utils.helpers import get_city_coordinates, parse_date
from utils.logger import log_tool_execution
from utils.validators import ValidationError, validate_city

TOOL_NAME = "weather_lookup_tool"

# WMO weather interpretation codes -> human-readable condition labels.
# Reference: https://open-meteo.com/en/docs (Weather variable documentation)
WMO_WEATHER_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def weather_code_to_condition(code: Optional[int]) -> str:
    """Translate a WMO weather code into a human-readable label."""
    if code is None:
        return "Unknown"
    return WMO_WEATHER_CODES.get(int(code), "Unknown")


class WeatherForecastInput(BaseModel):
    """Input schema for :func:`get_weather_forecast`."""

    city: str = Field(..., description="Destination city to fetch the forecast for")
    start_date: str = Field(..., description="First day of the trip, format YYYY-MM-DD")
    num_days: int = Field(
        default=settings.DEFAULT_TRIP_DAYS,
        ge=1,
        le=settings.WEATHER_FORECAST_DAYS,
        description="Number of consecutive days to fetch the forecast for",
    )


def _fetch_with_retry(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call ``url`` with ``params``, retrying transient failures.

    Implements a simple exponential backoff: each retry waits
    ``settings.WEATHER_RETRY_BACKOFF_SECONDS * attempt`` seconds.

    Raises:
        requests.RequestException: If all retry attempts fail.
    """
    last_exception: Optional[Exception] = None
    for attempt in range(1, settings.WEATHER_MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=settings.WEATHER_REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exception = exc
            if attempt < settings.WEATHER_MAX_RETRIES:
                backoff = settings.WEATHER_RETRY_BACKOFF_SECONDS * attempt
                time.sleep(backoff)
    assert last_exception is not None
    raise last_exception


@log_tool_execution(TOOL_NAME)
def get_weather_forecast(city: str, start_date: str, num_days: int = settings.DEFAULT_TRIP_DAYS) -> str:
    """Fetch a daily weather forecast for ``city`` starting on ``start_date``.

    Args:
        city: Destination city name (must be one of the supported cities
            with known coordinates).
        start_date: First day of the trip, ``YYYY-MM-DD``.
        num_days: Number of consecutive days to fetch (1 to
            ``settings.WEATHER_FORECAST_DAYS``).

    Returns:
        A JSON string encoding a :class:`models.response_models.ToolResponse`
        whose ``data`` field is a list of daily forecast dicts with keys
        ``date``, ``condition``, ``temperature_max_c``, ``temperature_min_c``
        and ``precipitation_probability_pct``.
    """
    try:
        city_name = validate_city(city, "city")
    except ValidationError as exc:
        return ToolResponse.fail(TOOL_NAME, error=str(exc)).to_json()

    try:
        start = parse_date(start_date)
    except ValueError:
        return ToolResponse.fail(
            TOOL_NAME, error=f"Invalid start_date '{start_date}'. Expected YYYY-MM-DD."
        ).to_json()

    coordinates = get_city_coordinates(city_name)
    if coordinates is None:
        return ToolResponse.fail(
            TOOL_NAME, error=f"No coordinates configured for city '{city_name}'."
        ).to_json()

    latitude, longitude = coordinates
    end = start + timedelta(days=num_days - 1)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
        "timezone": "auto",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }

    try:
        payload = _fetch_with_retry(settings.OPEN_METEO_BASE_URL, params)
    except requests.RequestException as exc:
        return ToolResponse.fail(
            TOOL_NAME,
            error=f"Failed to fetch weather data after {settings.WEATHER_MAX_RETRIES} attempts: {exc}",
        ).to_json()

    daily = payload.get("daily", {})
    dates: List[str] = daily.get("time", [])
    temps_max: List[float] = daily.get("temperature_2m_max", [])
    temps_min: List[float] = daily.get("temperature_2m_min", [])
    precipitation: List[float] = daily.get("precipitation_probability_max", [])
    codes: List[int] = daily.get("weathercode", [])

    forecast: List[Dict[str, Any]] = []
    for index, day in enumerate(dates):
        forecast.append(
            {
                "date": day,
                "condition": weather_code_to_condition(codes[index] if index < len(codes) else None),
                "temperature_max_c": temps_max[index] if index < len(temps_max) else None,
                "temperature_min_c": temps_min[index] if index < len(temps_min) else None,
                "precipitation_probability_pct": (
                    precipitation[index] if index < len(precipitation) else None
                ),
                "source": "open-meteo",
            }
        )

    if not forecast:
        return ToolResponse.ok(
            TOOL_NAME,
            data=[],
            message=(
                f"Open-Meteo returned no forecast data for {city_name} between "
                f"{params['start_date']} and {params['end_date']}. The dates may be "
                f"outside the available forecast range (typically up to "
                f"{settings.WEATHER_FORECAST_DAYS} days ahead)."
            ),
        ).to_json()

    return ToolResponse.ok(
        TOOL_NAME,
        data=forecast,
        message=f"Fetched {len(forecast)}-day forecast for {city_name} starting {start_date}.",
    ).to_json()


weather_lookup_tool = StructuredTool.from_function(
    func=get_weather_forecast,
    name=TOOL_NAME,
    description=(
        "Fetch a daily weather forecast (condition, max/min temperature in "
        "Celsius, and precipitation probability) for a destination city "
        "starting on a given date, using the free Open-Meteo API. Includes "
        "automatic retries on transient network errors. Returns a JSON "
        "ToolResponse containing the per-day forecast."
    ),
    args_schema=WeatherForecastInput,
)
