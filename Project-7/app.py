from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

# Ensure the project root is in the path when the app is launched from any cwd
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.workflow import run_trip_workflow
from agents.travel_agent import run_agent_chat
from configs import settings
from database.database import get_database
from models.user_request import TripRequest
from utils.validators import (
    SanitisationResult,
    ValidationError,
    sanitize_user_text,
    validate_city,
    validate_travel_style,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/travel-ai-assistant",
        "About": "Agentic AI Travel Planning Assistant — powered by LangChain & LangGraph",
    },
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stDateInput label { color: #94a3b8 !important; font-size: 0.85rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155; border-radius: 12px;
        padding: 1.2rem 1.5rem; color: #e2e8f0; text-align: center;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
    .metric-card .label { font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem; }
    .tool-badge {
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin: 0.15rem;
    }
    .badge-success { background: #14532d; color: #86efac; }
    .badge-error   { background: #7f1d1d; color: #fca5a5; }
    .stDownloadButton > button {
        background: #0369a1; color: #fff; border: none;
        border-radius: 8px; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CITIES = sorted(settings.CITY_COORDINATES.keys())
TRAVEL_STYLES = settings.TRAVEL_STYLES
TODAY = date.today()
DEFAULT_DEPARTURE = TODAY + timedelta(days=7)


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "page": "Trip Planner",
        "last_result": None,
        "history": [],
        "chat_messages": [],
        "session_id": f"ui-{TODAY.isoformat()}",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def _sidebar() -> str:
    with st.sidebar:
        st.markdown("## ✈️ AI Travel Planner")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["Trip Planner", "Travel Chat", "Past Itineraries", "Budget Analyser", "About"],
            index=["Trip Planner", "Travel Chat", "Past Itineraries", "Budget Analyser", "About"].index(
                st.session_state.page
            ),
        )
        st.markdown("---")
        st.markdown("**Quick Settings**")
        st.caption(f"Session: `{st.session_state.session_id}`")
        if st.button("🔄 New Session"):
            import uuid
            st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
            st.rerun()
        st.markdown("---")
        st.caption("Powered by LangChain · LangGraph · Streamlit")
    return page


# ---------------------------------------------------------------------------
# Page: Trip Planner
# ---------------------------------------------------------------------------
def _page_planner() -> None:
    st.title("🗺️ Plan Your Trip")
    st.markdown("Fill in the details below and let the AI agent build your personalised itinerary.")

    with st.form("planner_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("🛫 From City", CITIES, index=CITIES.index("Delhi") if "Delhi" in CITIES else 0)
            start_date = st.date_input("📅 Departure Date", value=DEFAULT_DEPARTURE, min_value=TODAY)
            budget = st.number_input(
                "💰 Total Budget (₹ INR)", min_value=5000, max_value=10_000_000,
                value=50_000, step=5_000
            )
        with col2:
            destination = st.selectbox(
                "🛬 To City", CITIES, index=CITIES.index("Goa") if "Goa" in CITIES else 1
            )
            num_days = st.number_input(
                "🗓️ Number of Days",
                min_value=settings.MIN_TRIP_DAYS,
                max_value=settings.MAX_TRIP_DAYS,
                value=settings.DEFAULT_TRIP_DAYS,
            )
            num_travellers = st.number_input("👥 Travellers", min_value=1, max_value=20, value=2)

        travel_style = st.selectbox("🎒 Travel Style", TRAVEL_STYLES)
        submitted = st.form_submit_button("✨ Generate Itinerary", use_container_width=True)

    if submitted:
        # Validation
        if source == destination:
            st.error("Source and destination cities must be different.")
            return

        with st.spinner("🤖 AI agents are planning your trip…"):
            try:
                trip_request = TripRequest(
                    source_city=source,
                    destination_city=destination,
                    start_date=start_date.isoformat(),
                    num_days=int(num_days),
                    budget=float(budget),
                    travel_style=travel_style,
                    num_travellers=int(num_travellers),
                )
            except (ValidationError, Exception) as exc:
                st.error(f"Invalid input: {exc}")
                return

            result = run_trip_workflow(trip_request, session_id=st.session_state.session_id)

        st.session_state.last_result = result
        if result.success:
            st.session_state.history.append(result)

        _display_result(result, source, destination, int(num_days), int(num_travellers))


def _display_result(result, source, destination, num_days, num_travellers):
    if not result.success:
        st.error(f"❌ Planning failed: {result.error}")
        st.markdown(result.final_answer)
        return

    st.success("✅ Itinerary generated successfully!")

    # --- KPI metrics ---
    itinerary_json = result.itinerary_json or {}
    budget_data = itinerary_json.get("budget", {})
    total_cost = budget_data.get("total_cost", 0)
    per_traveller = budget_data.get("per_traveller_cost", 0)
    budget_cat = budget_data.get("budget_category", "—")
    duration_s = result.duration_seconds

    cols = st.columns(4)
    kpis = [
        ("Total Cost", f"₹{total_cost:,.0f}", "INR"),
        ("Per Traveller", f"₹{per_traveller:,.0f}", "INR"),
        ("Budget Category", budget_cat, ""),
        ("Planning Time", f"{duration_s:.1f}s", "seconds"),
    ]
    for col, (label, value, unit) in zip(cols, kpis):
        col.markdown(
            f"<div class='metric-card'><div class='value'>{value}</div>"
            f"<div class='label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    tab_itin, tab_json, tab_trace = st.tabs(["📋 Itinerary", "🔗 JSON", "🔍 Agent Trace"])

    with tab_itin:
        src_coords = settings.CITY_COORDINATES.get(source)
        dst_coords = settings.CITY_COORDINATES.get(destination)
        if src_coords and dst_coords:
            import pandas as pd
            st.markdown("### 📍 Route Map")
            map_data = pd.DataFrame([
                {"lat": src_coords[0], "lon": src_coords[1], "city": source},
                {"lat": dst_coords[0], "lon": dst_coords[1], "city": destination},
            ])
            st.map(map_data)
            st.markdown("---")
            
        st.markdown(result.final_answer)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download Markdown",
                data=result.final_answer.encode("utf-8"),
                file_name=f"itinerary_{source}_to_{destination}_{num_days}days.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(itinerary_json, indent=2, ensure_ascii=False),
                file_name=f"itinerary_{source}_to_{destination}_{num_days}days.json",
                mime="application/json",
                use_container_width=True,
            )

    with tab_json:
        st.json(itinerary_json)

    with tab_trace:
        st.markdown("### 🔍 Agent Tool Trace")
        st.caption(f"Total planning time: **{duration_s:.2f}s** · Tools invoked: **{len(result.tool_calls)}**")
        for tc in result.tool_calls:
            status_cls = "badge-success" if tc.get("status") == "success" else "badge-error"
            status_label = tc.get("status", "unknown")
            ms = tc.get("duration_ms", 0)
            st.markdown(
                f"<span class='tool-badge {status_cls}'>{tc['tool']}</span> "
                f"<small style='color:#64748b;'>{ms:.1f} ms</small>",
                unsafe_allow_html=True,
            )
            if tc.get("error"):
                st.caption(f"  ⚠️ {tc['error']}")


# ---------------------------------------------------------------------------
# Page: Travel Chat
# ---------------------------------------------------------------------------
def _page_chat() -> None:
    st.title("💬 Travel Chat")
    st.markdown(
        "Ask the AI agent anything about Indian travel — flights, hotels, attractions, budget tips…"
    )

    # Render chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. What are the best places to visit in Goa?"):
        # Sanitise
        san: SanitisationResult = sanitize_user_text(prompt)
        if san.flagged:
            st.warning("⚠️ Your message was flagged for potential prompt-injection and was not sent.")
            return

        st.session_state.chat_messages.append({"role": "user", "content": san.text})
        with st.chat_message("user"):
            st.markdown(san.text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = run_agent_chat(san.text, session_id=st.session_state.session_id)
            st.markdown(result.final_answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": result.final_answer})


# ---------------------------------------------------------------------------
# Page: Past Itineraries
# ---------------------------------------------------------------------------
def _page_history() -> None:
    st.title("📚 Past Itineraries")
    db = get_database()
    itineraries = db.get_itineraries(limit=20)

    if not itineraries:
        st.info("No itineraries generated yet. Use the **Trip Planner** to create one!")
        return

    for row in itineraries:
        itin = row.get("itinerary_json") or {}
        title = row.get("trip_title", "Itinerary")
        cat = row.get("budget_category", "—")
        cost = row.get("total_cost", 0)
        created = row.get("created_at", "")
        with st.expander(f"🗺️ {title} — ₹{cost:,.0f} ({cat}) — {created[:10]}"):
            st.json(itin)
            json_str = json.dumps(itin, indent=2, ensure_ascii=False)
            st.download_button(
                "⬇️ Download JSON",
                data=json_str,
                file_name=f"{title.replace(' ', '_')}.json",
                mime="application/json",
                key=f"dl_{row['itinerary_id']}",
            )


# ---------------------------------------------------------------------------
# Page: Budget Analyser
# ---------------------------------------------------------------------------
def _page_budget() -> None:
    st.title("💰 Budget Analyser")
    st.markdown("Estimate trip cost before committing to a full plan.")

    col1, col2 = st.columns(2)
    with col1:
        flight_price = st.number_input("✈️ Flight Price per Person (₹)", min_value=0, value=5000, step=500)
        hotel_price = st.number_input("🏨 Hotel Price per Night (₹)", min_value=0, value=3000, step=500)
        num_days_b = st.number_input("🗓️ Number of Days", min_value=1, max_value=14, value=3)
    with col2:
        num_travellers_b = st.number_input("👥 Number of Travellers", min_value=1, max_value=20, value=2)
        user_budget_b = st.number_input("🎯 Your Target Budget (₹, optional)", min_value=0, value=0, step=1000)
        round_trip_b = st.checkbox("Round Trip?", value=True)

    if st.button("📊 Calculate", use_container_width=True):
        from tools.budget_tool import estimate_budget
        result_json = estimate_budget(
            flight_price_per_person=float(flight_price),
            hotel_price_per_night=float(hotel_price),
            num_days=int(num_days_b),
            num_travellers=int(num_travellers_b),
            round_trip=round_trip_b,
            user_budget=float(user_budget_b) if user_budget_b > 0 else None,
        )
        result = json.loads(result_json)
        if result["status"] != "success":
            st.error(result.get("error", "Unknown error"))
            return

        data = result["data"]
        st.success(result["message"])

        cols = st.columns(3)
        cols[0].metric("Total Cost", f"₹{data['total_cost']:,.0f}")
        cols[1].metric("Per Traveller", f"₹{data['per_traveller_cost']:,.0f}")
        cols[2].metric("Budget Category", data["budget_category"])

        st.markdown("### Cost Breakdown")
        breakdown_rows = {
            "✈️ Flights": data["flight_cost"],
            "🏨 Hotel": data["hotel_cost"],
            "🍽️ Food": data["food_cost"],
            "🚌 Local Transport": data["local_transport_cost"],
            "📦 Miscellaneous (10%)": data["miscellaneous_cost"],
        }
        for label, amount in breakdown_rows.items():
            pct = (amount / data["total_cost"] * 100) if data["total_cost"] else 0
            col_l, col_b, col_r = st.columns([2, 5, 1])
            col_l.markdown(label)
            col_b.progress(int(pct))
            col_r.markdown(f"₹{amount:,.0f}")

        if "within_budget" in data:
            if data["within_budget"]:
                st.success(f"✅ Within your budget! ₹{data['budget_difference']:,.0f} to spare.")
            else:
                st.error(f"❌ Exceeds your budget by ₹{abs(data['budget_difference']):,.0f}.")


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def _page_about() -> None:
    st.title("ℹ️ About This Project")
    st.markdown("""
## 🤖 Agentic AI Travel Planning Assistant

This project demonstrates an end-to-end **agentic AI** system for travel planning in India,
built with **LangChain**, **LangGraph**, and **Streamlit**.

### Architecture
```
User Query
    │
    ▼
LangGraph StateGraph Workflow
    ├── Intent Understanding (Planner Agent)
    ├── Flight Search Tool        (ranking engine)
    ├── Hotel Recommendation Tool (TF-IDF + FAISS semantic search)
    ├── Places Discovery Tool     (category + semantic filtering)
    ├── Weather Lookup Tool       (Open-Meteo API)
    ├── Budget Estimation Tool    (cost modelling)
    ├── Reasoning Layer           (LLM or template-based)
    └── Itinerary Builder         → SQLite persistence
```

### Dataset
- **30 flights** across 25 Indian city pairs
- **40 hotels** across 8 cities
- **40 tourist attractions** across 8 cities

### Cities Covered
`Bangalore · Chennai · Delhi · Goa · Hyderabad · Jaipur · Kolkata · Mumbai`

### Technology Stack
| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph StateGraph |
| LLM / Reasoning | LangChain + OpenAI GPT-4o-mini |
| Semantic Search | TF-IDF + FAISS (cosine similarity) |
| Web UI | Streamlit |
| Persistence | SQLite |
| Validation | Pydantic v2 |
| Testing | pytest (93 tests) |
| Logging | Loguru |
| Deployment | Docker + Docker Compose |
    """)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
page = _sidebar()
st.session_state.page = page

if page == "Trip Planner":
    _page_planner()
elif page == "Travel Chat":
    _page_chat()
elif page == "Past Itineraries":
    _page_history()
elif page == "Budget Analyser":
    _page_budget()
elif page == "About":
    _page_about()
