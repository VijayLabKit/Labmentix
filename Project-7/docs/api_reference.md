# API Reference

## Tools

### `flight_search_tool`
Search and rank available flights between two cities.

**Input schema**: `FlightSearchInput`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `str` | required | Departure city (must be one of the 8 supported cities) |
| `destination` | `str` | required | Arrival city (must differ from source) |
| `travel_date` | `str \| None` | `None` | Optional ISO date `YYYY-MM-DD` to filter by departure date |
| `criteria` | `"cheapest" \| "fastest" \| "best_value"` | `"best_value"` | Ranking strategy |
| `top_k` | `int` | `3` | Maximum results to return (1–20) |

**Response** (`ToolResponse[list[dict]]`): enriched flight records with `duration_minutes`, `duration_label`, `airline_rating`, and (for `best_value`) `value_score`.

---

### `hotel_recommendation_tool`
Search, filter, and semantically rank hotels in a given city.

**Input schema**: `HotelSearchInput`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `city` | `str` | required | City name |
| `max_price_per_night` | `float \| None` | `None` | Price ceiling (₹ per night) |
| `min_stars` | `int \| None` | `None` | Minimum star rating (1–5) |
| `sort_by` | `"price" \| "rating" \| "best_value"` | `"best_value"` | Ranking strategy |
| `preference_text` | `str \| None` | `None` | Free-text preference for semantic re-ranking |
| `top_k` | `int` | `5` | Maximum results (1–20) |

---

### `places_discovery_tool`
Discover tourist attractions in a city, optionally filtered by category.

**Input schema**: `PlaceSearchInput`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `city` | `str` | required | City name |
| `category` | `str \| None` | `None` | One of: `family`, `adventure`, `historical`, `cultural`, `relaxation` |
| `preference_text` | `str \| None` | `None` | Free-text preference for semantic re-ranking |
| `top_k` | `int` | `6` | Maximum results (1–40) |

---

### `weather_lookup_tool`
Fetch a multi-day weather forecast from Open-Meteo.

**Input schema**: `WeatherForecastInput`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `city` | `str` | required | City name |
| `start_date` | `str` | required | ISO date `YYYY-MM-DD` |
| `num_days` | `int` | `3` | Forecast days (1–7) |

**Response** (`ToolResponse[list[dict]]`): daily objects with `date`, `condition`, `temperature_max_c`, `temperature_min_c`, `precipitation_probability_pct`, `source`.

---

### `budget_estimation_tool`
Estimate the total cost of a trip.

**Input schema**: `BudgetEstimationInput`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `flight_price_per_person` | `float` | required | One-way price per person (₹) |
| `hotel_price_per_night` | `float` | required | Price per night (₹) |
| `num_days` | `int` | `3` | Trip length (1–14) |
| `num_travellers` | `int` | `1` | Number of travellers (1–20) |
| `round_trip` | `bool` | `True` | Double flight cost for return journey |
| `user_budget` | `float \| None` | `None` | Target budget for comparison |

**Cost model**:
```
flight_cost           = flight_price_per_person × multiplier × num_travellers
hotel_cost            = hotel_price_per_night × num_days × ceil(num_travellers / 2)
food_cost             = ₹900 × num_days × num_travellers
local_transport_cost  = ₹500 × num_days
miscellaneous_cost    = 10% of subtotal
total_cost            = subtotal + miscellaneous_cost
```

**Budget categories**: Budget (≤₹15,000/person) · Moderate (≤₹30,000) · Comfort (≤₹55,000) · Luxury (>₹55,000)

---

## Workflow Entry Point

### `run_trip_workflow(trip_request, session_id) → AgentRunResult`

**`TripRequest` fields**:

| Field | Type | Default | Description |
|---|---|---|---|
| `source_city` | `str` | required | Departure city |
| `destination_city` | `str` | required | Destination city |
| `start_date` | `date \| str` | required | Trip start date |
| `num_days` | `int` | `3` | Trip length (1–14) |
| `budget` | `float` | required | Total budget in ₹ |
| `travel_style` | `str` | `"Family"` | One of: Family, Adventure, Luxury, Backpacker |
| `num_travellers` | `int` | `1` | Number of travellers (1–20) |
| `raw_query` | `str \| None` | `None` | Original free-text query (optional) |

**`AgentRunResult` fields**:

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether the workflow completed successfully |
| `final_answer` | `str` | Markdown itinerary (or error message) |
| `itinerary_json` | `dict \| None` | Structured itinerary |
| `tool_calls` | `list[dict]` | Audit trail of tool invocations |
| `error` | `str \| None` | Error description if `success=False` |
| `duration_seconds` | `float` | Total workflow duration |

---

## ReAct Agent

### `answer_travel_question(question, session_id) → str`
Answer a free-text travel question using the LangChain ReAct agent (requires `OPENAI_API_KEY`). Falls back to a polite offline message if no key is configured.
