# Workflow Documentation

## LangGraph State Machine

The core of the assistant is a LangGraph `StateGraph` that moves a single `TravelPlanState` dict through nine sequential nodes:

```
start
  │
  ▼
[intent_understanding]
  │  Parses raw_query or validated TripRequest;
  │  derives flight search criteria (cheapest/fastest/best_value)
  │  and place category based on travel_style.
  │
  ▼
[flight_search]
  │  Calls flight_search_tool; selects top-ranked flight.
  │  Falls through gracefully if no flights exist on the route.
  │
  ▼
[hotel_search]
  │  Calls hotel_recommendation_tool with city + optional preference text.
  │  Selects the top-ranked hotel.
  │
  ▼
[places_search]
  │  Calls places_discovery_tool twice (style-specific category + generic).
  │  Merges and deduplicates results.
  │
  ▼
[weather_lookup]
  │  Calls weather_lookup_tool.
  │  Sets state["weather_warning"] and falls through if API unreachable.
  │
  ▼
[budget_estimation]
  │  Calls budget_estimation_tool with selected flight & hotel prices.
  │  If estimate exceeds user's budget, attempts one cheaper re-selection
  │  for both flight and hotel (single retry).
  │
  ▼
[reasoning]
  │  Generates a ReasoningTrace (flight / hotel / attractions /
  │  itinerary_ordering explanations) using the LLM when an API key
  │  is configured, or deterministic templates otherwise.
  │
  ▼
[build_itinerary]
  │  Calls services.itinerary_builder.build_itinerary().
  │  Produces a validated Itinerary Pydantic model.
  │  Generates Markdown via itinerary.to_markdown().
  │
  ▼
[persist]
  │  Saves query + itinerary + flight/hotel selections + all tool
  │  call logs to SQLite via database.TravelDatabase.
  │
  ▼
 END
```

## State Dictionary (`TravelPlanState`)

| Key | Type | Description |
|---|---|---|
| `trip_request` | `dict` | Serialised `TripRequest` |
| `session_id` | `str` | Client session identifier |
| `flights` | `list[dict]` | Ranked flight options |
| `selected_flight` | `dict \| None` | Chosen flight |
| `hotels` | `list[dict]` | Ranked hotel options |
| `selected_hotel` | `dict \| None` | Chosen hotel |
| `places` | `list[dict]` | Ranked attractions |
| `weather_forecast` | `list[dict]` | Daily weather objects |
| `budget_breakdown` | `dict \| None` | Cost breakdown |
| `reasoning` | `dict \| None` | Reasoning trace |
| `itinerary` | `dict \| None` | Final itinerary JSON |
| `itinerary_markdown` | `str \| None` | Human-readable output |
| `tool_calls` | `list[dict]` | Audit trail |
| `warnings` | `list[str]` | Non-fatal issues |
| `errors` | `list[str]` | Fatal issues |

## Error Handling & Graceful Degradation

| Scenario | Behaviour |
|---|---|
| No flights on requested route | Warning added; itinerary continues without flight |
| No hotels in destination | Warning added; itinerary continues without hotel |
| Open-Meteo unreachable | Warning added; weather section left empty |
| Budget exceeded | One re-selection attempt; warning if still over |
| No OpenAI API key | Template-based reasoning (fully offline) |
| Rate limit exceeded | Immediate `AgentRunResult(success=False, error="rate_limited")` |
| Unexpected exception | Caught at workflow level; graceful error response |

## Security Controls

- **Prompt injection detection**: `sanitize_user_text()` scans input for known injection patterns before processing.
- **Rate limiting**: Sliding-window `RateLimiter` (30 requests / 60 seconds per session) applied at workflow entry.
- **Input validation**: All city names, budgets, trip durations and travel styles are validated through `utils.validators` before the workflow begins.
