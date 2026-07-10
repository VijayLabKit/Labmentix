"""
Security and validation utilities.

This module implements the three "Security" requirements that are not
covered elsewhere:

    1. Input validation for user-supplied trip requests.
    2. Prompt-injection detection/sanitisation for free-text fields that
       are interpolated into LLM prompts.
    3. A simple in-memory rate limiter to protect the agent endpoint from
       abuse.

All three are deliberately framework-agnostic (no Streamlit / LangChain
imports) so they can be reused from the API layer, the Streamlit UI, or
unit tests.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List

from configs import settings
from utils.helpers import normalise_city
from utils.logger import get_logger

logger = get_logger("validators")


class ValidationError(ValueError):
    """Raised when user-supplied input fails validation."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

VALID_CITIES = set(settings.CITY_COORDINATES.keys())
VALID_TRAVEL_STYLES = {style.lower() for style in settings.TRAVEL_STYLES}


def validate_city(city: str, field_name: str = "city") -> str:
    """Validate and normalise a city name against the supported city list.

    Args:
        city: Raw city name supplied by the user.
        field_name: Name of the field, used in error messages.

    Returns:
        The normalised (title-cased) city name.

    Raises:
        ValidationError: If the city is empty or not one of the supported
            cities present in the underlying datasets.
    """
    if not city or not city.strip():
        raise ValidationError(f"{field_name} must not be empty.")

    normalised = normalise_city(city)
    if normalised not in VALID_CITIES:
        supported = ", ".join(sorted(VALID_CITIES))
        raise ValidationError(
            f"{field_name} '{city}' is not supported. Supported cities: {supported}."
        )
    return normalised


def validate_trip_duration(days: int) -> int:
    """Validate the requested trip length in days.

    Raises:
        ValidationError: If ``days`` is outside the configured bounds.
    """
    if not isinstance(days, int) or isinstance(days, bool):
        raise ValidationError("Trip duration must be an integer number of days.")
    if days < settings.MIN_TRIP_DAYS or days > settings.MAX_TRIP_DAYS:
        raise ValidationError(
            f"Trip duration must be between {settings.MIN_TRIP_DAYS} and "
            f"{settings.MAX_TRIP_DAYS} days (got {days})."
        )
    return days


def validate_budget(budget: float) -> float:
    """Validate that a budget value is a positive, finite number."""
    try:
        budget_value = float(budget)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Budget must be a numeric value.") from exc

    if budget_value <= 0:
        raise ValidationError("Budget must be greater than zero.")
    if budget_value > 10_000_000:
        raise ValidationError("Budget value is unrealistically large.")
    return budget_value


def validate_travel_style(style: str) -> str:
    """Validate a travel style against the supported list."""
    if not style:
        raise ValidationError("Travel style must not be empty.")
    normalised = style.strip().lower()
    if normalised not in VALID_TRAVEL_STYLES:
        supported = ", ".join(settings.TRAVEL_STYLES)
        raise ValidationError(
            f"Travel style '{style}' is not supported. Supported styles: {supported}."
        )
    return normalised.title()


def validate_source_destination(source: str, destination: str) -> tuple:
    """Validate a source/destination pair, ensuring they differ."""
    source_city = validate_city(source, "source city")
    destination_city = validate_city(destination, "destination city")
    if source_city == destination_city:
        raise ValidationError("Source and destination cities must be different.")
    return source_city, destination_city


# ---------------------------------------------------------------------------
# Prompt injection protection
# ---------------------------------------------------------------------------

# Patterns commonly used in prompt-injection attempts. This is a defence-in-
# depth measure: it does not try to be exhaustive, but it catches the most
# common "ignore previous instructions" style attacks and strips characters
# that are frequently used to break out of an instruction block.
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore (all|any|previous|the) (instructions|prompts|rules)", re.I),
    re.compile(r"disregard (all|any|previous|the) (instructions|prompts|rules)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"act as (an?|the)", re.I),
    re.compile(r"reveal (your|the) (system|hidden) prompt", re.I),
    re.compile(r"</?(system|assistant|user)>", re.I),
    re.compile(r"```"),
]

_MAX_FREE_TEXT_LENGTH = 1000


@dataclass
class SanitisationResult:
    """Result of sanitising a free-text field."""

    text: str
    flagged: bool
    matched_patterns: List[str]


def sanitize_user_text(text: str, field_name: str = "input") -> SanitisationResult:
    """Sanitise free-text user input before it is interpolated into a prompt.

    The function:
        * Truncates overly long input.
        * Strips characters often used to escape an instruction block
          (backticks, angle-bracket pseudo-tags).
        * Flags (but does not silently execute) text that matches known
          prompt-injection patterns so callers can log/alert.

    Args:
        text: Raw user-supplied text.
        field_name: Name of the field, used for logging.

    Returns:
        A :class:`SanitisationResult` containing the cleaned text plus
        metadata about any suspicious patterns that were detected.
    """
    if text is None:
        return SanitisationResult(text="", flagged=False, matched_patterns=[])

    original = str(text)
    truncated = original[:_MAX_FREE_TEXT_LENGTH]

    matched: List[str] = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(truncated):
            matched.append(pattern.pattern)

    cleaned = truncated.replace("```", "").replace("<system>", "").replace("</system>", "")
    cleaned = cleaned.replace("<assistant>", "").replace("</assistant>", "")

    if matched:
        logger.warning(
            "Potential prompt injection detected in field '{}': patterns={}",
            field_name,
            matched,
        )

    return SanitisationResult(text=cleaned.strip(), flagged=bool(matched), matched_patterns=matched)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """A simple in-memory sliding-window rate limiter.

    Suitable for single-process deployments (e.g. a Streamlit app or a
    single Uvicorn worker). For multi-worker / multi-instance deployments
    this should be backed by Redis or a similar shared store.
    """

    def __init__(
        self,
        max_requests: int = settings.RATE_LIMIT_MAX_REQUESTS,
        window_seconds: int = settings.RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, client_id: str) -> bool:
        """Return ``True`` if ``client_id`` is within their rate limit.

        As a side effect, records the current request timestamp when the
        request is allowed.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds
        history = self._requests[client_id]

        while history and history[0] < window_start:
            history.popleft()

        if len(history) >= self.max_requests:
            logger.warning(
                "Rate limit exceeded for client '{}': {} requests in {}s window",
                client_id,
                len(history),
                self.window_seconds,
            )
            return False

        history.append(now)
        return True

    def remaining(self, client_id: str) -> int:
        """Return the number of requests remaining in the current window."""
        now = time.monotonic()
        window_start = now - self.window_seconds
        history = self._requests[client_id]
        while history and history[0] < window_start:
            history.popleft()
        return max(self.max_requests - len(history), 0)


# Shared, module-level limiter instance used by the agent entry points.
global_rate_limiter = RateLimiter()
