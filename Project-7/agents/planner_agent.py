"""
Planner Agent -- the "Intent Understanding" layer.

Converts a free-text user query (e.g. *"Plan a 4 day family trip from Delhi
to Goa starting 2026-07-10, budget 60000 for 2 people"*) into a validated
:class:`models.user_request.TripRequest`.

Two extraction strategies are supported:

    1. **Heuristic extraction** (:func:`heuristic_extract`) -- regex- and
       keyword-based parsing that requires no external services. This is
       always available and is used as the default / fallback strategy so
       the application remains fully functional offline.
    2. **LLM-assisted extraction** (:func:`llm_extract`) -- when
       ``settings.OPENAI_API_KEY`` is configured, a ``ChatOpenAI`` model is
       used with structured output to fill in any fields the heuristic
       parser could not confidently determine.

:func:`extract_trip_request` combines both strategies and returns a tuple of
``(TripRequest | None, warnings, errors)``.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from configs import settings
from models.user_request import TripRequest
from utils.logger import get_logger
from utils.validators import sanitize_user_text

logger = get_logger("planner_agent")

_KNOWN_CITIES = sorted(settings.CITY_COORDINATES.keys())
_CITY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(city) for city in _KNOWN_CITIES) + r")\b",
    re.IGNORECASE,
)
_FROM_TO_PATTERN = re.compile(
    r"from\s+([A-Za-z ]+?)\s+(?:to|->|towards)\s+([A-Za-z ]+?)(?=[,.\n]|$| on | for | with | starting | from)",
    re.IGNORECASE,
)
_DAYS_PATTERN = re.compile(r"(\d+)\s*[- ]?\s*(?:day|days|night|nights)", re.IGNORECASE)
_ISO_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_TRAVELLERS_PATTERN = re.compile(
    r"(\d+)\s*(?:people|persons?|travell?ers?|adults?|pax|members?)", re.IGNORECASE
)
_BUDGET_PATTERN = re.compile(
    r"(?:budget|inr|rs\.?|₹)\s*(?:of|is|around|about|approx\.?|approximately)?\s*"
    r"[:\-]?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|lakh|lac)?",
    re.IGNORECASE,
)
_BUDGET_SUFFIX_PATTERN = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(k|thousand|lakh|lac)?\s*(?:rupees|rs\.?|inr|₹)", re.IGNORECASE
)
_BUDGET_FALLBACK_PATTERN = re.compile(
    r"\b([\d,]{4,})\s*(k|thousand|lakh|lac)?\b", re.IGNORECASE
)

_MULTIPLIERS = {"k": 1_000, "thousand": 1_000, "lakh": 100_000, "lac": 100_000}


class HeuristicExtraction(BaseModel):
    """Container for fields recovered by :func:`heuristic_extract`."""

    source_city: Optional[str] = None
    destination_city: Optional[str] = None
    start_date: Optional[date] = None
    num_days: Optional[int] = None
    budget: Optional[float] = None
    travel_style: Optional[str] = None
    num_travellers: Optional[int] = None


def _normalise_city_token(token: str) -> Optional[str]:
    token = token.strip()
    for city in _KNOWN_CITIES:
        if city.lower() == token.lower():
            return city
    return None


def _extract_cities(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Find a source/destination city pair in free text.

    First tries an explicit ``"from X to Y"`` pattern; if that fails, falls
    back to the first two distinct known city names mentioned, in the order
    they appear.
    """
    match = _FROM_TO_PATTERN.search(text)
    if match:
        source = _normalise_city_token(match.group(1))
        destination = _normalise_city_token(match.group(2))
        if source and destination and source != destination:
            return source, destination

    seen: List[str] = []
    for m in _CITY_PATTERN.finditer(text):
        city = _normalise_city_token(m.group(1))
        if city and city not in seen:
            seen.append(city)
        if len(seen) == 2:
            break

    source = seen[0] if len(seen) >= 1 else None
    destination = seen[1] if len(seen) >= 2 else None
    return source, destination


def _extract_start_date(text: str) -> Optional[date]:
    match = _ISO_DATE_PATTERN.search(text)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _extract_num_days(text: str) -> Optional[int]:
    match = _DAYS_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_num_travellers(text: str) -> Optional[int]:
    match = _TRAVELLERS_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_amount(raw_amount: str, suffix: Optional[str]) -> float:
    amount = float(raw_amount.replace(",", ""))
    if suffix:
        amount *= _MULTIPLIERS.get(suffix.lower(), 1)
    return amount


def _extract_budget(text: str) -> Optional[float]:
    match = _BUDGET_PATTERN.search(text)
    if match:
        try:
            return _parse_amount(match.group(1), match.group(2))
        except ValueError:
            pass

    match = _BUDGET_SUFFIX_PATTERN.search(text)
    if match:
        try:
            return _parse_amount(match.group(1), match.group(2))
        except ValueError:
            pass

    # Fallback: any standalone 4+ digit number, optionally with a k/lakh
    # suffix, that is not part of a date or a "N day(s)" phrase.
    for m in _BUDGET_FALLBACK_PATTERN.finditer(text):
        span_text = text[max(0, m.start() - 6) : m.end() + 6].lower()
        if "day" in span_text or "night" in span_text or _ISO_DATE_PATTERN.search(m.group(0)):
            continue
        try:
            return _parse_amount(m.group(1), m.group(2))
        except ValueError:
            continue
    return None


def _extract_travel_style(text: str) -> Optional[str]:
    lowered = text.lower()
    for style in settings.TRAVEL_STYLES:
        if style.lower() in lowered:
            return style
    return None


def heuristic_extract(query: str) -> HeuristicExtraction:
    """Extract trip-planning fields from free text using regex/keyword rules.

    This function never raises -- any field that cannot be confidently
    extracted is left as ``None`` so the caller can apply defaults or
    request clarification.
    """
    source_city, destination_city = _extract_cities(query)
    return HeuristicExtraction(
        source_city=source_city,
        destination_city=destination_city,
        start_date=_extract_start_date(query),
        num_days=_extract_num_days(query),
        budget=_extract_budget(query),
        travel_style=_extract_travel_style(query),
        num_travellers=_extract_num_travellers(query),
    )


class _LLMTripFields(BaseModel):
    """Schema used to request structured trip fields from the LLM."""

    source_city: Optional[str] = Field(default=None, description="Departure city")
    destination_city: Optional[str] = Field(default=None, description="Destination city")
    start_date: Optional[str] = Field(default=None, description="Trip start date, YYYY-MM-DD")
    num_days: Optional[int] = Field(default=None, description="Trip length in days")
    budget: Optional[float] = Field(default=None, description="Total trip budget in INR")
    travel_style: Optional[str] = Field(
        default=None, description="One of Family, Adventure, Luxury, Backpacker"
    )
    num_travellers: Optional[int] = Field(default=None, description="Number of travellers")


def llm_extract(query: str) -> Optional[_LLMTripFields]:
    """Use an LLM with structured output to extract trip fields.

    Returns ``None`` (without raising) if no API key is configured or if
    the LLM call fails for any reason -- callers should fall back to
    :func:`heuristic_extract` in that case.
    """
    if not settings.GEMINI_API_KEY:
        return None

    try:
        from utils.llm_provider import get_llm

        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(_LLMTripFields)
        prompt = (
            "Extract trip-planning details from the following user request. "
            "Only the cities Delhi, Mumbai, Bangalore, Chennai, Kolkata, "
            "Hyderabad, Goa and Jaipur are supported -- leave a field as null "
            "if it is not mentioned or not one of these cities. travel_style "
            "must be one of Family, Adventure, Luxury or Backpacker if "
            f"mentioned.\n\nUser request: {query}"
        )
        result = structured_llm.invoke(prompt)
        if isinstance(result, _LLMTripFields):
            return result
        return _LLMTripFields(**result)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - any LLM/network failure is non-fatal here
        logger.warning("LLM-assisted intent extraction failed, using heuristics only: {}", exc)
        return None


def extract_trip_request(
    query: str, session_id: str = "default"
) -> Tuple[Optional[TripRequest], List[str], List[str]]:
    """Build a :class:`TripRequest` from a free-text query.

    Args:
        query: The user's natural-language request.
        session_id: Session identifier, used only for logging.

    Returns:
        A tuple ``(trip_request, warnings, errors)``. ``trip_request`` is
        ``None`` if mandatory fields (source city, destination city and
        budget) could not be determined; in that case ``errors`` explains
        what is missing. ``warnings`` lists any fields that were filled in
        with defaults.
    """
    sanitised = sanitize_user_text(query, field_name="planner_query")
    if sanitised.flagged:
        logger.warning(
            "Sanitised planner query for session '{}' flagged patterns: {}",
            session_id,
            sanitised.matched_patterns,
        )
    text = sanitised.text

    heuristics = heuristic_extract(text)
    llm_fields = llm_extract(text)

    def _pick(field: str) -> Any:
        heuristic_value = getattr(heuristics, field)
        if heuristic_value is not None:
            return heuristic_value
        if llm_fields is not None:
            return getattr(llm_fields, field, None)
        return None

    source_city = _pick("source_city")
    destination_city = _pick("destination_city")
    start_date_value = heuristics.start_date
    if start_date_value is None and llm_fields and llm_fields.start_date:
        try:
            start_date_value = date.fromisoformat(llm_fields.start_date)
        except ValueError:
            start_date_value = None
    num_days = _pick("num_days")
    budget = _pick("budget")
    travel_style = _pick("travel_style")
    num_travellers = _pick("num_travellers")

    warnings: List[str] = []
    errors: List[str] = []

    if not source_city or not destination_city:
        errors.append(
            "Could not determine the source and destination cities. Supported "
            f"cities are: {', '.join(_KNOWN_CITIES)}."
        )
        return None, warnings, errors

    if budget is None:
        errors.append(
            "Could not determine a trip budget. Please mention an amount in "
            "INR, e.g. 'budget 50000' or '50k'."
        )
        return None, warnings, errors

    if start_date_value is None:
        start_date_value = date.today() + timedelta(days=7)
        warnings.append(
            f"No start date detected -- defaulting to {start_date_value.isoformat()} "
            "(one week from today)."
        )

    if num_days is None:
        num_days = settings.DEFAULT_TRIP_DAYS
        warnings.append(f"No trip duration detected -- defaulting to {num_days} day(s).")

    if travel_style is None:
        travel_style = "Family"
        warnings.append("No travel style detected -- defaulting to 'Family'.")

    if num_travellers is None:
        num_travellers = 1
        warnings.append("No traveller count detected -- defaulting to 1 traveller.")

    try:
        trip_request = TripRequest(
            source_city=source_city,
            destination_city=destination_city,
            start_date=start_date_value,
            num_days=int(num_days),
            budget=float(budget),
            travel_style=travel_style,
            num_travellers=int(num_travellers),
            raw_query=text,
        )
    except (PydanticValidationError, ValueError) as exc:
        errors.append(str(exc))
        return None, warnings, errors

    logger.info(
        "Extracted trip request for session '{}': {} -> {}, {} day(s), style={}",
        session_id,
        trip_request.source_city,
        trip_request.destination_city,
        trip_request.num_days,
        trip_request.travel_style,
    )
    return trip_request, warnings, errors
