"""
Pydantic models describing the user's trip planning request.

These models are the single entry point for user input -- both the
Streamlit UI and the LangChain agent build a :class:`TripRequest` instance,
which performs validation via Pydantic validators backed by
``utils.validators``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from configs import settings
from utils.validators import (
    ValidationError,
    validate_budget,
    validate_source_destination,
    validate_travel_style,
    validate_trip_duration,
)


class TripRequest(BaseModel):
    """A validated request to plan a trip.

    Attributes:
        source_city: City the traveller departs from.
        destination_city: City the traveller is travelling to.
        start_date: First day of the trip (``YYYY-MM-DD``).
        num_days: Length of the trip in days (inclusive of arrival day).
        budget: Total budget for the trip, in INR, for the whole party.
        travel_style: One of ``Family``, ``Adventure``, ``Luxury``,
            ``Backpacker`` -- used to bias hotel/attraction selection.
        num_travellers: Number of people travelling (affects hotel cost
            assumptions and per-person budget reporting).
        raw_query: The original free-text query from the user, if the
            request originated from a natural-language prompt. This field
            is sanitised separately before being passed to the LLM.
    """

    source_city: str = Field(..., description="Departure city")
    destination_city: str = Field(..., description="Destination city")
    start_date: date = Field(..., description="Trip start date (YYYY-MM-DD)")
    num_days: int = Field(
        default=settings.DEFAULT_TRIP_DAYS,
        ge=settings.MIN_TRIP_DAYS,
        le=settings.MAX_TRIP_DAYS,
        description="Number of days for the trip",
    )
    budget: float = Field(..., gt=0, description="Total trip budget in INR")
    travel_style: str = Field(default="Family", description="Preferred travel style")
    num_travellers: int = Field(default=1, ge=1, le=20, description="Number of travellers")
    raw_query: Optional[str] = Field(
        default=None, description="Original free-text query, if any"
    )

    @field_validator("source_city", "destination_city")
    @classmethod
    def _strip_city(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("City must not be empty")
        return value.strip()

    @field_validator("travel_style")
    @classmethod
    def _normalise_style(cls, value: str) -> str:
        try:
            return validate_travel_style(value)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("budget")
    @classmethod
    def _check_budget(cls, value: float) -> float:
        try:
            return validate_budget(value)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _check_cities_and_duration(self) -> "TripRequest":
        try:
            source, destination = validate_source_destination(
                self.source_city, self.destination_city
            )
            validate_trip_duration(self.num_days)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

        self.source_city = source
        self.destination_city = destination
        return self

    @property
    def end_date(self) -> date:
        """Return the trip's end (last day) date, inclusive."""
        from datetime import timedelta

        return self.start_date + timedelta(days=self.num_days - 1)

    def to_prompt_context(self) -> str:
        """Render the request as a short natural-language summary for the agent."""
        return (
            f"Plan a {self.num_days}-day trip from {self.source_city} to "
            f"{self.destination_city}, starting {self.start_date.isoformat()}, "
            f"for {self.num_travellers} traveller(s), with a total budget of "
            f"INR {self.budget:,.0f}, in the '{self.travel_style}' travel style."
        )


class AgentQuery(BaseModel):
    """Wraps a free-text user query before it is handed to the agent."""

    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", description="Client/session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
