# ✈️ Agentic AI Travel Planning Assistant

An end-to-end **agentic AI** application for personalised Indian travel planning, built with **LangChain**, **LangGraph**, and **Streamlit**.

---

## 📸 Screenshots

> _Run the app with `streamlit run app.py` and capture screenshots into `docs/screenshots/`._

| Trip Planner | Generated Itinerary | Budget Analyser |
|:---:|:---:|:---:|
| ![Planner](docs/screenshots/01_trip_planner_form.png) | ![Itinerary](docs/screenshots/02_itinerary_output.png) | ![Budget](docs/screenshots/06_budget_analyser.png) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- (Optional) OpenAI API key for LLM reasoning & Travel Chat

### Installation

```powershell
# 1. Open PowerShell and go to the project folder
cd e:\Project-7

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Create the environment file and add your API key (optional)
copy .env.example .env
notepad .env
# add OPENAI_API_KEY=sk-...

# 6. Launch the app
streamlit run app.py
```

Visit **http://localhost:8501** in your browser.

### Docker

```bash
# Build and run
docker compose up --build

# Run tests only
docker compose run --rm tests
```

---

## 🗺️ Features

| Feature | Description |
|---|---|
| **Trip Planner** | Fill a form → AI generates a full multi-day itinerary |
| **Travel Chat** | Free-text Q&A via a LangChain ReAct agent |
| **Past Itineraries** | Browse and download previously generated plans |
| **Budget Analyser** | Standalone cost estimator with breakdown chart |
| **Semantic Search** | TF-IDF + FAISS cosine similarity for hotels & attractions |
| **Offline Mode** | Full itinerary generation without an OpenAI API key |
| **Security** | Prompt-injection detection · Rate limiting · Input validation |

---

## 🏙️ Covered Cities

`Bangalore · Chennai · Delhi · Goa · Hyderabad · Jaipur · Kolkata · Mumbai`

---

## 📂 Project Structure

```
travel_ai_assistant/
├── app.py                      # Streamlit web application
├── agents/
│   ├── workflow.py             # LangGraph StateGraph pipeline
│   ├── travel_agent.py         # LangChain ReAct tool-calling agent
│   └── planner_agent.py        # Intent parser
├── tools/
│   ├── flight_tool.py          # Flight search & ranking
│   ├── hotel_tool.py           # Hotel recommendation
│   ├── place_tool.py           # Attraction discovery
│   ├── weather_tool.py         # Open-Meteo weather lookup
│   └── budget_tool.py          # Cost estimation
├── services/
│   ├── ranking_engine.py       # Multi-criteria ranking
│   ├── recommendation_engine.py# TF-IDF + FAISS semantic search
│   └── itinerary_builder.py    # Itinerary assembly
├── models/
│   ├── user_request.py         # TripRequest Pydantic model
│   ├── itinerary.py            # Itinerary + sub-models
│   └── response_models.py      # ToolResponse, AgentRunResult
├── database/
│   ├── schema.sql              # SQLite schema
│   └── database.py             # TravelDatabase class
├── utils/
│   ├── helpers.py              # Data loading, parsing, normalisation
│   ├── validators.py           # Input validation, injection guard, rate limiter
│   └── logger.py               # Loguru structured logging
├── configs/
│   └── settings.py             # Central configuration
├── data/
│   ├── flights.json            # 30 flights
│   ├── hotels.json             # 40 hotels
│   └── places.json             # 40 attractions
├── tests/                      # 93 pytest tests
├── docs/                       # Architecture, workflow, API reference
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
# 93 tests pass fully offline (weather tests mock HTTP; agent tests mock the LLM)
```

---

## ⚙️ Configuration

All settings are in `configs/settings.py` and can be overridden via environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model |
| `OPENAI_TEMPERATURE` | `0.2` | Generation temperature |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `RATE_LIMIT_MAX_REQUESTS` | `30` | Max requests per window |
| `WEATHER_FORECAST_DAYS` | `7` | Max weather forecast days |

---

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Workflow Documentation](docs/workflow.md)
- [API Reference](docs/api_reference.md)

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph 1.2 StateGraph |
| LLM / Reasoning | LangChain 1.3 + OpenAI GPT-4o-mini |
| Semantic Search | scikit-learn TF-IDF + FAISS 1.14 |
| Web UI | Streamlit 1.58 |
| Data Validation | Pydantic v2 |
| Persistence | SQLite (stdlib) |
| Logging | Loguru 0.7 |
| Testing | pytest 9 (93 tests) |
| Deployment | Docker + Docker Compose |

---

## 📝 License

MIT — see [LICENSE](LICENSE) for details.
