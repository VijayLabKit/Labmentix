# Architecture Overview

## System Architecture

The Agentic AI Travel Planning Assistant is structured as a multi-layer application:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI (app.py)                    │
│  Trip Planner │ Travel Chat │ Past Itineraries │ Budget Analyser │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              LangGraph StateGraph Workflow                       │
│          (agents/workflow.py)                                   │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Intent  │  │  Flight  │  │  Hotel   │  │  Places  │       │
│  │ Planner  │→ │  Search  │→ │  Search  │→ │Discovery │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                  │               │
│                    ┌─────────┐  ┌─────────┐     │               │
│                    │ Weather │  │ Budget  │←────┘               │
│                    │ Lookup  │→ │  Est.   │                     │
│                    └─────────┘  └─────────┘                     │
│                                    │                             │
│                    ┌───────────────▼──────────────┐            │
│                    │  Reasoning + Itinerary Build  │            │
│                    └───────────────┬──────────────┘            │
└───────────────────────────────────┼─────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────┐
│                  Tool Layer (tools/)                             │
│  flight_tool  hotel_tool  place_tool  weather_tool  budget_tool │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────┐
│               Service Layer (services/)                         │
│    ranking_engine    recommendation_engine    itinerary_builder  │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
┌────────────────────┬──────────────▼──────────────┬─────────────┐
│   Data Layer       │   Database (SQLite)          │   Models    │
│   (data/*.json)    │   (database/database.py)     │  (models/)  │
└────────────────────┴─────────────────────────────┴─────────────┘
```

## Component Descriptions

### Streamlit UI (`app.py`)
The multi-page web interface providing:
- **Trip Planner**: Form-based itinerary generation
- **Travel Chat**: Free-text Q&A via the ReAct agent
- **Past Itineraries**: Browse and download previously generated plans
- **Budget Analyser**: Standalone cost estimation tool

### LangGraph Workflow (`agents/workflow.py`)
A deterministic `StateGraph` that orchestrates the full planning pipeline. Each node is a pure function operating on `TravelPlanState`. The workflow requires no LLM API key to run — reasoning falls back to template-based explanations, and weather degrades gracefully when the API is unreachable.

### Planner Agent (`agents/planner_agent.py`)
Parses free-text queries to extract structured intent (source/destination/dates/budget/style) using either the LLM (when configured) or regex-based extraction.

### ReAct Travel Agent (`agents/travel_agent.py`)
A LangChain tool-calling agent available via the Travel Chat page. Autonomously selects and invokes the five tools to answer open-ended travel questions.

### Tool Layer (`tools/`)
Five `StructuredTool` LangChain tools, each with a Pydantic input schema:

| Tool | Description |
|---|---|
| `flight_search_tool` | Filter + rank flights by criteria |
| `hotel_recommendation_tool` | Filter + semantic-rank hotels |
| `places_discovery_tool` | Filter + semantic-rank attractions |
| `weather_lookup_tool` | Fetch 7-day forecast from Open-Meteo |
| `budget_estimation_tool` | Compute full cost breakdown |

### Service Layer (`services/`)
- **`ranking_engine`**: Deterministic multi-criteria ranking (price, duration, rating, value-score) for flights, hotels, and places.
- **`recommendation_engine`**: TF-IDF + FAISS (or NumPy fallback) semantic search over hotel and attraction descriptions.
- **`itinerary_builder`**: Assembles all components into a validated `Itinerary` Pydantic model and generates Markdown output.

### Data Layer (`data/`)
Three JSON datasets: `flights.json` (30 records), `hotels.json` (40 records), `places.json` (40 records) covering 8 Indian cities.

### Database (`database/`)
SQLite persistence for user queries, generated itineraries, flight/hotel selections, and tool-call logs. Schema in `schema.sql`.

### Models (`models/`)
Pydantic v2 models: `TripRequest`, `Itinerary`, `BudgetBreakdown`, `FlightOption`, `HotelOption`, `AttractionOption`, `DailyWeather`, `DayPlan`, `ToolResponse`, `AgentRunResult`.

### Utilities (`utils/`)
- **`logger`**: Loguru-based structured logging with file rotation.
- **`helpers`**: Data loading, datetime parsing, normalisation, categorisation.
- **`validators`**: Input validation, prompt-injection detection, rate limiting.
