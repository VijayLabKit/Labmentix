"""
Budget Estimation Tool.

Given a selected flight price, a selected hotel's per-night price, the
number of days/nights of the trip and the number of travellers, this tool
produces a full cost breakdown that mirrors
:class:`models.itinerary.BudgetBreakdown` exactly, so its output dict can be
passed straight into ``BudgetBreakdown(**data)`` inside
:func:`services.itinerary_builder.build_itinerary`.

The estimation model is intentionally simple and transparent:

    * Flight cost  = ``flight_price`` (assumed round-trip-equivalent price
      already present in the dataset, multiplied by the number of
      travellers).
    * Hotel cost   = ``hotel_price_per_night`` x number of nights
      (``num_days``), independent of traveller count (one room is assumed
      per booking unless the caller scales ``hotel_price_per_night`` up
      front).
    * Food cost    = ``settings.DAILY_FOOD_COST_INR`` x ``num_days`` x
      ``num_travellers``.
    * Local transport cost = ``settings.DAILY_LOCAL_TRANSPORT_COST_INR`` x
      ``num_days`` x ``num_travellers``.
    * Miscellaneous cost = ``settings.MISCELLANEOUS_RATE`` x (flight + hotel
      + food + transport) subtotal.
    * Total cost   = sum of all of the above.
    * Daily budget = ``total_cost / num_days``.
    * Budget category is derived from ``total_cost`` per traveller against
      :data:`configs.settings.BUDGET_CATEGORY_THRESHOLDS`.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from configs import settings
from models.response_models import ToolResponse
from utils.helpers import safe_round
from utils.logger import log_tool_execution

TOOL_NAME = "budget_estimation_tool"


class BudgetEstimationInput(BaseModel):
    """Input schema for :func:`estimate_budget`."""

    flight_price: float = Field(
        ..., ge=0, description="Price of the selected flight per traveller, in INR."
    )
    hotel_price_per_night: float = Field(
        ..., ge=0, description="Price per night of the selected hotel, in INR."
    )
    num_days: int = Field(
        default=settings.DEFAULT_TRIP_DAYS,
        ge=settings.MIN_TRIP_DAYS,
        le=settings.MAX_TRIP_DAYS,
        description="Number of days (and nights) of the trip.",
    )
    num_travellers: int = Field(
        default=1, ge=1, le=20, description="Number of people travelling on this trip."
    )


def _classify_budget(total_cost_per_traveller: float) -> str:
    """Map a per-traveller total cost to a budget category label.

    Iterates :data:`configs.settings.BUDGET_CATEGORY_THRESHOLDS` in
    ascending order of threshold and returns the first category whose
    threshold is greater than or equal to the supplied cost.
    """
    ordered = sorted(settings.BUDGET_CATEGORY_THRESHOLDS.items(), key=lambda item: item[1])
    for category, threshold in ordered:
        if total_cost_per_traveller <= threshold:
            return category
    # Fallback - should never happen since the last threshold is inf.
    return ordered[-1][0]


@log_tool_execution(TOOL_NAME)
def estimate_budget(
    flight_price: float,
    hotel_price_per_night: float,
    num_days: int = settings.DEFAULT_TRIP_DAYS,
    num_travellers: int = 1,
) -> str:
    """Estimate a full trip cost breakdown.

    Args:
        flight_price: Price of the selected flight per traveller (INR).
        hotel_price_per_night: Price per night of the selected hotel (INR).
        num_days: Number of days/nights of the trip.
        num_travellers: Number of travellers.

    Returns:
        A JSON string encoding a :class:`models.response_models.ToolResponse`
        whose ``data`` field is a dictionary with the keys
        ``flight_cost``, ``hotel_cost``, ``food_cost``,
        ``local_transport_cost``, ``miscellaneous_cost``, ``total_cost``,
        ``daily_budget``, ``budget_category``, ``currency``,
        ``num_travellers`` and ``per_traveller_cost`` -- matching
        :class:`models.itinerary.BudgetBreakdown`.
    """
    if num_days <= 0:
        return ToolResponse.fail(TOOL_NAME, error="num_days must be a positive integer").to_json()
    if num_travellers <= 0:
        return ToolResponse.fail(
            TOOL_NAME, error="num_travellers must be a positive integer"
        ).to_json()

    flight_cost = flight_price * num_travellers
    hotel_cost = hotel_price_per_night * num_days
    food_cost = settings.DAILY_FOOD_COST_INR * num_days * num_travellers
    local_transport_cost = settings.DAILY_LOCAL_TRANSPORT_COST_INR * num_days * num_travellers

    subtotal = flight_cost + hotel_cost + food_cost + local_transport_cost
    miscellaneous_cost = subtotal * settings.MISCELLANEOUS_RATE
    total_cost = subtotal + miscellaneous_cost

    daily_budget = total_cost / num_days
    per_traveller_cost = total_cost / num_travellers
    budget_category = _classify_budget(per_traveller_cost)

    breakdown: Dict[str, Any] = {
        "flight_cost": safe_round(flight_cost),
        "hotel_cost": safe_round(hotel_cost),
        "food_cost": safe_round(food_cost),
        "local_transport_cost": safe_round(local_transport_cost),
        "miscellaneous_cost": safe_round(miscellaneous_cost),
        "total_cost": safe_round(total_cost),
        "daily_budget": safe_round(daily_budget),
        "budget_category": budget_category,
        "currency": "INR",
        "num_travellers": num_travellers,
        "per_traveller_cost": safe_round(per_traveller_cost),
    }

    return ToolResponse.ok(
        TOOL_NAME,
        data=breakdown,
        message=(
            f"Estimated total trip cost is INR {breakdown['total_cost']:,.2f} "
            f"({breakdown['budget_category']} category) for {num_travellers} "
            f"traveller(s) over {num_days} day(s)."
        ),
    ).to_json()


budget_estimation_tool = StructuredTool.from_function(
    func=estimate_budget,
    name=TOOL_NAME,
    description=(
        "Estimate a full cost breakdown for a trip given a selected flight price "
        "(per traveller), a selected hotel's price per night, the number of "
        "days/nights, and the number of travellers. Adds estimated daily food "
        "and local transport costs plus a miscellaneous buffer, and classifies "
        "the trip into a budget category (Budget, Moderate, Comfort or Luxury). "
        "Returns a JSON ToolResponse with the full cost breakdown."
    ),
    args_schema=BudgetEstimationInput,
)
