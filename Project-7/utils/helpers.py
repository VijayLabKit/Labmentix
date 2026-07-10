"""
General-purpose helper functions used across tools, services and agents.

This module intentionally has no dependencies on LangChain so it can be
imported and unit-tested in isolation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from configs import settings

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"


def load_json_dataset(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSON array dataset (flights/hotels/places) from disk.

    The function is deliberately tolerant of dataset size -- it streams the
    file via the standard ``json`` module which is efficient for the
    medium-sized arrays used by this project (tens to tens of thousands of
    records) without requiring any schema changes as the dataset grows.

    Args:
        path: Path to a JSON file containing a top-level array of objects.

    Returns:
        A list of dictionaries. Returns an empty list if the file does not
        exist (callers should treat this as "no data available" rather than
        a fatal error so the UI can degrade gracefully).

    Raises:
        ValueError: If the file exists but does not contain a JSON array.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {file_path}, got {type(data)}")

    return data


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 style datetime string used in flights.json."""
    return datetime.strptime(value, DATETIME_FORMAT)


def parse_date(value: str) -> datetime:
    """Parse a ``YYYY-MM-DD`` date string."""
    return datetime.strptime(value, DATE_FORMAT)


def flight_duration_minutes(departure_time: str, arrival_time: str) -> int:
    """Compute a flight's duration in minutes from ISO timestamps.

    Handles overnight flights where the arrival timestamp's date is after
    the departure timestamp's date.
    """
    departure = parse_datetime(departure_time)
    arrival = parse_datetime(arrival_time)
    delta = arrival - departure
    if delta.total_seconds() < 0:
        # Defensive guard for malformed data -- treat as same-day.
        delta = timedelta(seconds=abs(delta.total_seconds()))
    return int(delta.total_seconds() // 60)


def format_duration(minutes: int) -> str:
    """Format a duration given in minutes as ``"Xh Ym"``."""
    hours, mins = divmod(max(minutes, 0), 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins}m"


def normalise_city(city: str) -> str:
    """Normalise a city name for case-insensitive comparisons."""
    return city.strip().title()


def get_city_coordinates(city: str) -> Optional[tuple]:
    """Return ``(latitude, longitude)`` for a known city, or ``None``."""
    return settings.CITY_COORDINATES.get(normalise_city(city))


def get_airline_rating(airline: str) -> float:
    """Return the curated rating for an airline, with a sensible default."""
    return settings.AIRLINE_RATINGS.get(airline, settings.DEFAULT_AIRLINE_RATING)


def categorise_place(place_type: str) -> List[str]:
    """Map a raw place ``type`` to one or more travel-style categories."""
    return settings.PLACE_TYPE_CATEGORY_MAP.get(place_type.lower(), ["Cultural"])


def min_max_normalise(value: float, minimum: float, maximum: float) -> float:
    """Normalise ``value`` to the ``[0, 1]`` range given known bounds.

    Returns ``0.5`` if ``minimum == maximum`` to avoid a divide-by-zero and
    to keep all candidates on equal footing when there is no variance.
    """
    if maximum == minimum:
        return 0.5
    return (value - minimum) / (maximum - minimum)


def daterange(start_date: datetime, num_days: int) -> List[datetime]:
    """Return a list of ``num_days`` consecutive dates starting at ``start_date``."""
    return [start_date + timedelta(days=offset) for offset in range(num_days)]


def chunked(iterable: Iterable, size: int) -> Iterable[list]:
    """Yield successive ``size``-sized chunks from ``iterable``."""
    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


_WHITESPACE_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space and strip the result."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def safe_round(value: float, ndigits: int = 2) -> float:
    """Round a float while tolerating ``None``/non-numeric input (returns 0.0)."""
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return 0.0
