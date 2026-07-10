"""
Central configuration module for the Agentic AI Travel Planning Assistant.

All tunables, static reference data, and environment-driven settings are
defined here so the rest of the application has a single source of truth.
Values that should change between environments (API keys, model names,
file paths) are read from environment variables via `python-dotenv`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load variables from a .env file if present (no-op in environments where
# the variables are already provided, e.g. Docker / CI).
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean flag from the environment."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer value from the environment."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float value from the environment."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# General application metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "Agentic AI Travel Planning Assistant"
APP_VERSION: str = "1.0.0"
ENVIRONMENT: str = os.getenv("APP_ENV", "development")
DEBUG: bool = _env_bool("DEBUG", default=False)

# ---------------------------------------------------------------------------
# Data file locations
# ---------------------------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
FLIGHTS_FILE: Path = Path(os.getenv("FLIGHTS_FILE", str(DATA_DIR / "flights.json")))
HOTELS_FILE: Path = Path(os.getenv("HOTELS_FILE", str(DATA_DIR / "hotels.json")))
PLACES_FILE: Path = Path(os.getenv("PLACES_FILE", str(DATA_DIR / "places.json")))

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------
DATABASE_DIR: Path = BASE_DIR / "database"
DATABASE_PATH: Path = Path(os.getenv("DATABASE_PATH", str(DATABASE_DIR / "travel_assistant.db")))
SCHEMA_PATH: Path = DATABASE_DIR / "schema.sql"

# ---------------------------------------------------------------------------
# LLM / LangChain configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE: float = _env_float("GEMINI_TEMPERATURE", 0.3)
GEMINI_MAX_OUTPUT_TOKENS: int = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 4096)
AGENT_MAX_ITERATIONS: int = _env_int("AGENT_MAX_ITERATIONS", 8)
AGENT_VERBOSE: bool = _env_bool("AGENT_VERBOSE", default=True)

# ---------------------------------------------------------------------------
# Open-Meteo weather API configuration
# ---------------------------------------------------------------------------
OPEN_METEO_BASE_URL: str = os.getenv(
    "OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1/forecast"
)
WEATHER_REQUEST_TIMEOUT: int = _env_int("WEATHER_REQUEST_TIMEOUT", 10)
WEATHER_MAX_RETRIES: int = _env_int("WEATHER_MAX_RETRIES", 3)
WEATHER_RETRY_BACKOFF_SECONDS: float = _env_float("WEATHER_RETRY_BACKOFF_SECONDS", 1.5)
WEATHER_FORECAST_DAYS: int = _env_int("WEATHER_FORECAST_DAYS", 7)

# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory token bucket, see utils.validators)
# ---------------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS: int = _env_int("RATE_LIMIT_MAX_REQUESTS", 30)
RATE_LIMIT_WINDOW_SECONDS: int = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_DIR: Path = BASE_DIR / "logs"
LOG_FILE: Path = LOG_DIR / "travel_assistant.log"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_ROTATION: str = os.getenv("LOG_ROTATION", "5 MB")
LOG_RETENTION: str = os.getenv("LOG_RETENTION", "10 days")

# ---------------------------------------------------------------------------
# FAISS / semantic recommendation configuration
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR: Path = BASE_DIR / "database" / "faiss_index"
FAISS_HOTELS_INDEX: Path = FAISS_INDEX_DIR / "hotels.index"
FAISS_PLACES_INDEX: Path = FAISS_INDEX_DIR / "places.index"
EMBEDDING_DIMENSIONS: int = _env_int("EMBEDDING_DIMENSIONS", 256)

# ---------------------------------------------------------------------------
# Trip / budget defaults
# ---------------------------------------------------------------------------
DEFAULT_TRIP_DAYS: int = _env_int("DEFAULT_TRIP_DAYS", 3)
MAX_TRIP_DAYS: int = _env_int("MAX_TRIP_DAYS", 14)
MIN_TRIP_DAYS: int = 1

# Per-day cost assumptions (in INR) used by the Budget Estimation Tool.
DAILY_FOOD_COST_INR: float = _env_float("DAILY_FOOD_COST_INR", 900.0)
DAILY_LOCAL_TRANSPORT_COST_INR: float = _env_float("DAILY_LOCAL_TRANSPORT_COST_INR", 500.0)
MISCELLANEOUS_RATE: float = _env_float("MISCELLANEOUS_RATE", 0.10)  # 10% of subtotal

# Budget category thresholds (total trip cost, in INR, per traveller).
BUDGET_CATEGORY_THRESHOLDS: Dict[str, float] = {
    "Budget": 15000.0,
    "Moderate": 30000.0,
    "Comfort": 55000.0,
    "Luxury": float("inf"),
}

# Number of attractions to schedule per day (Morning / Afternoon / Evening).
SLOTS_PER_DAY: List[str] = ["Morning", "Afternoon", "Evening"]

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

# Approximate latitude/longitude for the cities present in the supplied
# datasets. Used by the Weather Tool to query Open-Meteo, which requires
# coordinates rather than city names.
CITY_COORDINATES: Dict[str, Tuple[float, float]] = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Goa": (15.2993, 74.1240),
    "Jaipur": (26.9124, 75.7873),
}

# Airline quality ratings (out of 5). These are not present in flights.json
# so a curated reference table is used to feed the "best value" ranking
# score (price + duration + airline rating).
AIRLINE_RATINGS: Dict[str, float] = {
    "IndiGo": 4.2,
    "Air India": 3.8,
    "SpiceJet": 3.6,
    "Vistara": 4.6,
    "Go First": 3.4,
}
DEFAULT_AIRLINE_RATING: float = 3.5

# Mapping of raw "place type" values found in places.json to the broader
# travel-style categories used for filtering (Family / Adventure /
# Historical / Cultural / Relaxation).
PLACE_TYPE_CATEGORY_MAP: Dict[str, List[str]] = {
    "beach": ["Adventure", "Family", "Relaxation"],
    "fort": ["Historical", "Cultural"],
    "lake": ["Relaxation", "Family"],
    "market": ["Family", "Cultural"],
    "monument": ["Historical", "Cultural"],
    "museum": ["Cultural", "Historical"],
    "park": ["Family", "Relaxation", "Adventure"],
    "temple": ["Cultural", "Historical"],
}

TRAVEL_STYLES: List[str] = ["Family", "Adventure", "Luxury", "Backpacker"]

# Ranking weights used by services.ranking_engine for the "best value"
# flight score. Lower price/duration is better, higher airline rating is
# better, so the score normalises each component to [0, 1] before
# combining them.
FLIGHT_RANKING_WEIGHTS: Dict[str, float] = {
    "price": 0.5,
    "duration": 0.3,
    "airline_rating": 0.2,
}

HOTEL_RANKING_WEIGHTS: Dict[str, float] = {
    "price": 0.4,
    "rating": 0.6,
}


def ensure_directories() -> None:
    """Create runtime directories that the application writes to."""
    for directory in (LOG_DIR, DATABASE_DIR, FAISS_INDEX_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
