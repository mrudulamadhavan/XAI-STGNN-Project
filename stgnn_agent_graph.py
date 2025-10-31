from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable

def ui_agent(state: dict) -> dict:
    print("[UIAgent] User input received.")
    state["user_query"] = "Predict traffic and suggest alternate route"
    return state

def forecasting_agent(state: dict) -> dict:
    print("[ForecastingAgent] Running dummy STGNN model...")
    state["forecast"] = "Heavy congestion in Zone A expected in 20 minutes"
    return state

def knowledge_agent(state: dict) -> dict:
    print("[KnowledgeAgent] Fetching similar historical traffic events...")
    state["similar_events"] = ["Event-X on Jan 12", "Event-Y on Feb 8"]
    return state

def recommendation_agent(state: dict) -> dict:
    print("[RecommendationAgent] Generating alternate route...")
    state["recommendation"] = "Use Route B → C → D (less congested)"
    return state

def explanation_agent(state: dict) -> dict:
    print("[ExplanationAgent] Creating final explanation...")
    state["explanation"] = (
        f"{state['forecast']}. Based on past events: {', '.join(state['similar_events'])}. "
        f"Recommended: {state['recommendation']}"
    )
    return state

def output_agent(state: dict) -> dict:
    print("\n[UIAgent Output] Final Response to User:")
    print(state["explanation"])
    return state

# === Build LangGraph ===
def build_graph():
    builder = StateGraph(dict)

    # Add function-based nodes
    builder.add_node("UIAgent", ui_agent)
    builder.add_node("ForecastingAgent", forecasting_agent)
    builder.add_node("KnowledgeAgent", knowledge_agent)
    builder.add_node("RecommendationAgent", recommendation_agent)
    builder.add_node("ExplanationAgent", explanation_agent)
    builder.add_node("OutputAgent", output_agent)

    # Transitions
    builder.set_entry_point("UIAgent")
    builder.add_edge("UIAgent", "ForecastingAgent")
    builder.add_edge("ForecastingAgent", "KnowledgeAgent")
    builder.add_edge("KnowledgeAgent", "RecommendationAgent")
    builder.add_edge("RecommendationAgent", "ExplanationAgent")
    builder.add_edge("ExplanationAgent", "OutputAgent")
    builder.add_edge("OutputAgent", END)

    return builder.compile()

# === Run ===
if __name__ == "__main__":
    graph = build_graph()
    final_state = graph.invoke({})