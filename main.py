# main.py
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode

from math_agent import math_agent, math_tools, math_tool_node
from optimiser_agent import optimiser_agent, optimiser_tools
from supervisor_agent import supervisor_agent


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "continue" if getattr(last, "tool_calls", None) else "end"


def route_from_state(state: AgentState) -> str:
    return state.get("route", "math_agent")


def build_langgraph_app():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("math_agent", math_agent)
    graph.add_node("optimiser_agent", optimiser_agent)

    graph.add_node("math_tools", ToolNode(math_tools))
    graph.add_node("optimiser_tools", ToolNode(optimiser_tools))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_state,
        {
            "math_agent": "math_agent",
            "optimiser_agent": "optimiser_agent",
        },
    )

    graph.add_conditional_edges(
        "math_agent",
        should_continue,
        {"continue": "math_tools", "end": END},
    )
    graph.add_edge("math_tools", "math_agent")

    graph.add_conditional_edges(
        "optimiser_agent",
        should_continue,
        {"continue": "optimiser_tools", "end": END},
    )
    graph.add_edge("optimiser_tools", "optimiser_agent")

    return graph.compile()


def run():
    app = build_langgraph_app()

    state = {"messages": [], "route": "math_agent"}
    while True:
        user = input("You: ").strip()
        if user in {"exit", "quit"}:
            break

        state["messages"].append(HumanMessage(content=user))
        for s in app.stream(state, stream_mode="values"):
            msg = s["messages"][-1]
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                print("🧠", msg.content)
            state = s


if __name__ == "__main__":
    run()
