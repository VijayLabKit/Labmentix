"""
Pydantic models describing the structured itinerary produced by the
Itinerary Builder service and returned by the agent.

These models double as the JSON schema for the "Download JSON" feature in
the Streamlit UI and as the contract between ``services.itinerary_builder``
and the presentation layer.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import List, Optional

from pydantic import BaseModel, Field


class FlightOption(BaseModel):
    """A single flight candidate, optionally enriched with ranking info."""

    flight_id: str
    airline: str
    source: str = Field(alias="from")
    destination: str = Field(alias="to")
    departure_time: str
    arrival_time: str
    price: float
    duration_minutes: int
    duration_label: str
    airline_rating: float
    value_score: Optional[float] = None

    model_config = {"populate_by_name": True}


class HotelOption(BaseModel):
    """A single hotel candidate, optionally enriched with ranking info."""

    hotel_id: str
    name: str
    city: str
    stars: int
    price_per_night: float
    amenities: List[str] = Field(default_factory=list)
    value_score: Optional[float] = None
    semantic_score: Optional[float] = None


class AttractionOption(BaseModel):
    """A single tourist attraction / place of interest."""

    place_id: str
    name: str
    city: str
    type: str
    categories: List[str] = Field(default_factory=list)
    rating: float
    semantic_score: Optional[float] = None


class DailyWeather(BaseModel):
    """Weather forecast for a single calendar day."""

    date: str
    condition: str
    temperature_max_c: Optional[float] = None
    temperature_min_c: Optional[float] = None
    precipitation_probability_pct: Optional[float] = None
    source: str = "open-meteo"


class DaySlotActivity(BaseModel):
    """A single Morning/Afternoon/Evening activity within a day plan."""

    slot: str
    place: Optional[AttractionOption] = None
    notes: Optional[str] = None


class DayPlan(BaseModel):
    """A single day's plan within the itinerary."""

    day_number: int
    date: str
    weather: Optional[DailyWeather] = None
    activities: List[DaySlotActivity] = Field(default_factory=list)


class BudgetBreakdown(BaseModel):
    """Cost breakdown for the whole trip."""

    flight_cost: float
    hotel_cost: float
    food_cost: float
    local_transport_cost: float
    miscellaneous_cost: float
    total_cost: float
    daily_budget: float
    budget_category: str
    currency: str = "INR"
    num_travellers: int = 1
    per_traveller_cost: Optional[float] = None


class ReasoningTrace(BaseModel):
    """Transparent reasoning behind each major recommendation."""

    flight_reasoning: str
    hotel_reasoning: str
    attractions_reasoning: str
    itinerary_ordering_reasoning: str


class Itinerary(BaseModel):
    """The complete, structured trip itinerary."""

    trip_title: str
    source_city: str
    destination_city: str
    start_date: str
    end_date: str
    num_days: int
    travel_style: str
    selected_flight: Optional[FlightOption] = None
    selected_hotel: Optional[HotelOption] = None
    weather_forecast: List[DailyWeather] = Field(default_factory=list)
    day_plans: List[DayPlan] = Field(default_factory=list)
    budget: Optional[BudgetBreakdown] = None
    reasoning: Optional[ReasoningTrace] = None
    travel_tips: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render a human-readable Markdown summary of the itinerary."""
        lines: List[str] = []
        
        # Executive Summary
        lines.append("# Executive Summary")
        lines.append(f"This is a {self.num_days}-day {self.travel_style.lower()} trip from {self.source_city} to {self.destination_city}.")
        lines.append(f"The trip is scheduled from {self.start_date} to {self.end_date}.")
        if self.budget:
            lines.append(f"Total estimated cost is ₹{self.budget.total_cost:,.0f} ({self.budget.budget_category}).")
        lines.append("")

        # Day-by-Day Itinerary
        lines.append("# Day-by-Day Itinerary")
        if self.day_plans:
            for day_plan in self.day_plans:
                lines.append(f"### Day {day_plan.day_number} — {day_plan.date}")
                if day_plan.weather:
                    lines.append(f"_Weather: {day_plan.weather.condition}_")
                for activity in day_plan.activities:
                    if activity.place:
                        lines.append(f"- **{activity.slot}:** {activity.place.name} ({activity.place.type.title()}, rating {activity.place.rating})")
                    elif activity.notes:
                        lines.append(f"- **{activity.slot}:** {activity.notes}")
                lines.append("")
        else:
            lines.append("No specific day plans available.\n")

        # Attractions
        lines.append("# Attractions")
        if self.reasoning and self.reasoning.attractions_reasoning:
            lines.append(self.reasoning.attractions_reasoning)
        else:
            lines.append("Attractions are distributed throughout the day-by-day itinerary.")
        lines.append("")

        # Transportation
        lines.append("# Transportation")
        if self.selected_flight:
            f = self.selected_flight
            lines.append(f"**Flight:** {f.airline} ({f.flight_id}) from {f.source} to {f.destination}.")
            lines.append(f"Departure: {f.departure_time} | Arrival: {f.arrival_time} | Duration: {f.duration_label}")
            lines.append(f"Price: ₹{f.price:,.0f} | Airline rating: {f.airline_rating}/5")
        else:
            lines.append("No flights selected.")
        if self.reasoning and self.reasoning.flight_reasoning:
            lines.append(f"_{self.reasoning.flight_reasoning}_")
        lines.append("")

        # Budget Breakdown
        lines.append("# Budget Breakdown")
        if self.budget:
            b = self.budget
            lines.append(f"- Flight cost: ₹{b.flight_cost:,.0f}")
            lines.append(f"- Hotel cost: ₹{b.hotel_cost:,.0f}")
            lines.append(f"- Food cost: ₹{b.food_cost:,.0f}")
            lines.append(f"- Local transport: ₹{b.local_transport_cost:,.0f}")
            lines.append(f"- Miscellaneous: ₹{b.miscellaneous_cost:,.0f}")
            lines.append(f"- **Total cost: ₹{b.total_cost:,.0f}**")
            lines.append(f"- Daily budget: ₹{b.daily_budget:,.0f}")
            lines.append(f"- Budget category: **{b.budget_category}**")
        else:
            lines.append("No budget breakdown available.")
        lines.append("")

        # Food Recommendations
        lines.append("# Food Recommendations")
        food_cost = self.budget.food_cost if self.budget else 0
        lines.append(f"Consider exploring local dining options in {self.destination_city} that fit a {self.travel_style.lower()} budget. Allow roughly ₹{food_cost:,.0f} for meals.")
        lines.append("")

        # Weather Notes
        lines.append("# Weather Notes")
        if self.weather_forecast:
            for w in self.weather_forecast:
                temp = f", {w.temperature_min_c:.0f}°C - {w.temperature_max_c:.0f}°C" if w.temperature_max_c and w.temperature_min_c else ""
                lines.append(f"- {w.date}: {w.condition}{temp}")
        else:
            lines.append("No weather data available.")
        lines.append("")

        # Packing Suggestions
        lines.append("# Packing Suggestions")
        lines.append("Pack according to the expected weather conditions and your chosen travel style. Ensure you have comfortable footwear for sightseeing.")
        lines.append("")

        # Safety Information
        lines.append("# Safety Information")
        lines.append("Always keep your belongings secure and stay aware of your surroundings in crowded tourist spots. Keep emergency contacts handy.")
        lines.append("")

        # Local Tips
        lines.append("# Local Tips")
        for tip in self.travel_tips:
            lines.append(f"- {tip}")

        return "\n".join(lines)
