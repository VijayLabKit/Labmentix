import json
import time
from models.user_request import TripRequest
from agents.workflow import run_trip_workflow

request = TripRequest(
    source_city="Delhi",
    destination_city="Goa",
    start_date="2026-06-25",
    num_days=3,
    budget=50000,
    travel_style="Family",
    num_travellers=2
)

print("Running workflow...")
result = run_trip_workflow(request, "test-session")
print(f"Success: {result.success}")
print(f"Error: {result.error}")
print("Tool Calls:")
for tc in result.tool_calls:
    print(f"- {tc['tool']}: status={tc.get('status')} duration_ms={tc.get('duration_ms')} error={tc.get('error')}")

