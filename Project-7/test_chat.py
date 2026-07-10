from agents.travel_agent import run_agent_chat

result = run_agent_chat("What are the best places to visit in Goa?")
print(f"Success: {result.success}")
print(f"Final Answer: {result.final_answer}")
print(f"Error: {result.error}")
print("Tool Calls:")
for tc in result.tool_calls:
    print(tc)
