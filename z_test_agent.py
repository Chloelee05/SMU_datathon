from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

import z_data_in
import pandas as pd
import numpy as np
import difflib
import heapq



load_dotenv()

# -----------------------------
# Load data
# -----------------------------
cargill_vessels, cargill_cargoes, market_vessels, market_cargoes, port_distances, ffa, bunker = z_data_in.load_data()


# -----------------------------
# Agent state
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# One-time caches (vessel_name unique, cargo_id unique)
# -----------------------------
def _norm(s: str) -> str:
    return str(s).strip().casefold()


def _norm_port(s: str) -> str:
    return str(s).strip().upper()


# Vessels: unique vessel_name across combined set
_ALL_VESSELS = pd.concat(
    [cargill_vessels.assign(_src="cargill"), market_vessels.assign(_src="market")],
    ignore_index=True,
).copy()
_ALL_VESSELS["_name_key"] = _ALL_VESSELS["vessel_name"].astype(str).map(_norm)
_VESSEL_BY_NAME = _ALL_VESSELS.set_index("_name_key", drop=False)


# Cargoes: unique cargo_id across combined set
_ALL_CARGOES = pd.concat(
    [cargill_cargoes.assign(_src="cargill"), market_cargoes.assign(_src="market")],
    ignore_index=True,
).copy()
_ALL_CARGOES["_id_key"] = _ALL_CARGOES["cargo_id"].astype(str).map(_norm)
_CARGO_BY_ID = _ALL_CARGOES.set_index("_id_key", drop=False)


# --- enforce uniqueness assumptions early ---
dupe_v = _ALL_VESSELS["_name_key"].duplicated(keep=False)
if dupe_v.any():
    dupes = _ALL_VESSELS.loc[dupe_v, ["vessel_name", "_src"]].sort_values("vessel_name")
    raise ValueError(f"Duplicate vessel_name keys found:\n{dupes.to_string(index=False)}")

dupe_c = _ALL_CARGOES["_id_key"].duplicated(keep=False)
if dupe_c.any():
    dupes = _ALL_CARGOES.loc[dupe_c, ["cargo_id", "_src"]].sort_values("cargo_id")
    raise ValueError(f"Duplicate cargo_id keys found:\n{dupes.to_string(index=False)}")


# Distances: symmetric map (A,B)->nm for O(1) lookup
_pd = port_distances.copy()
_pd["A"] = _pd["PORT_NAME_FROM"].astype(str).map(_norm_port)
_pd["B"] = _pd["PORT_NAME_TO"].astype(str).map(_norm_port)
_pd["D"] = pd.to_numeric(_pd["DISTANCE"], errors="coerce")


_DISTANCE_MAP: dict[tuple[str, str], float] = {}
for a, b, d in zip(_pd["A"], _pd["B"], _pd["D"]):
    if pd.notna(d):
        _DISTANCE_MAP[(a, b)] = float(d)
        _DISTANCE_MAP[(b, a)] = float(d)  # symmetric


# Deterministic list of valid ports for fuzzy matching / suggestions
_VALID_PORTS = sorted({p for (a, b) in _DISTANCE_MAP.keys() for p in (a, b)})

_GRAPH: dict[str, list[tuple[str, float]]] = {}
for a, b, d in zip(_pd["A"], _pd["B"], _pd["D"]):
    if pd.notna(d):
        _GRAPH.setdefault(a, []).append((b, float(d)))
        _GRAPH.setdefault(b, []).append((a, float(d)))


def _similarity(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(a=a, b=b).ratio())

def _nearest_port(port: str, *, cutoff: float = 0.80) -> tuple[str | None, float]:
    p = _norm_port(port)
    if p in _VALID_PORTS:
        return p, 1.0
    matches = difflib.get_close_matches(p, _VALID_PORTS, n=1, cutoff=float(cutoff))
    if not matches:
        return None, 0.0
    best = matches[0]
    return best, _similarity(p, best)

def _shortest_path_distance(start: str, goal: str, *, max_hops: int = 6) -> tuple[float | None, list[str] | None]:
    """
    Dijkstra shortest path. Returns (total_distance, route_ports) or (None, None).
    max_hops prevents crazy multi-hop assumptions.
    """
    start = _norm_port(start)
    goal = _norm_port(goal)
    if start == goal:
        return 0.0, [start]

    dist: dict[str, float] = {start: 0.0}
    prev: dict[str, str] = {}
    hops: dict[str, int] = {start: 0}

    pq: list[tuple[float, str]] = [(0.0, start)]
    visited: set[str] = set()

    while pq:
        dcur, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            route = [goal]
            while route[-1] in prev:
                route.append(prev[route[-1]])
            route.reverse()
            return dcur, route

        for nxt, w in _GRAPH.get(node, []):
            nh = hops.get(node, 0) + 1
            if nh > max_hops:
                continue
            nd = dcur + float(w)
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = node
                hops[nxt] = nh
                heapq.heappush(pq, (nd, nxt))

    return None, None




# -----------------------------
# Tools
# -----------------------------
@tool
def check_vessel_exists(vessel_name: str) -> bool:
    """
    Checks if a vessel exists in our database (Cargill + Market).
    Input: vessel_name
    Output: bool
    """
    if not isinstance(vessel_name, str) or not vessel_name.strip():
        return False
    return _norm(vessel_name) in _VESSEL_BY_NAME.index


@tool
def check_cargo_exists(cargo_id: str) -> bool:
    """
    Checks if a cargo exists in our database (Cargill + Market).
    Input: cargo_id
    Output: bool
    """
    if not isinstance(cargo_id, str) or not cargo_id.strip():
        return False
    return _norm(cargo_id) in _CARGO_BY_ID.index


@tool
def suggest_ports(query: str, k: int = 5) -> dict:
    """
    Suggest closest known port names from the distance table.

    Inputs:
      - query: port string
      - k: number of suggestions (default 5, max 10)

    Output:
      {
        "ok": bool,
        "query": str,
        "normalized": str,
        "suggestions": [{"port": str, "score": float}, ...],
        "error": str | None
      }
    """
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "query": query, "normalized": "", "suggestions": [], "error": "Empty query"}

    try:
        kk = int(k)
    except Exception:
        kk = 5
    kk = max(1, min(10, kk))

    q = _norm_port(query)

    # Exact match
    if q in _VALID_PORTS:
        return {
            "ok": True,
            "query": query,
            "normalized": q,
            "suggestions": [{"port": q, "score": 1.0}],
            "error": None,
        }

    candidates = difflib.get_close_matches(q, _VALID_PORTS, n=kk, cutoff=0.0)
    scored = [{"port": c, "score": _similarity(q, c)} for c in candidates]
    scored.sort(key=lambda x: x["score"], reverse=True)

    return {"ok": True, "query": query, "normalized": q, "suggestions": scored[:kk], "error": None}


@tool
def get_distance_nm(port_a: str, port_b: str, cutoff: float = 0.80, max_hops: int = 6) -> dict:
    """
    Use distance table provided. If direct distance missing, make assumptions:
      - route via intermediate ports (shortest path over table)
      - nearest port substitution if port not found
    Never invents distances outside the table; only sums known legs.

    Returns:
      {
        "ok": bool,
        "distance_nm": float|None,
        "port_a_used": str|None,
        "port_b_used": str|None,
        "matched": bool,
        "score_a": float,
        "score_b": float,
        "method": "direct"|"routed"|"fuzzy_routed"|None,
        "route_ports": list[str]|None,
        "assumptions": list[str],
        "error": str|None
      }
    """
    assumptions: list[str] = []

    if not all(isinstance(x, str) and x.strip() for x in (port_a, port_b)):
        return {"ok": False, "distance_nm": None, "port_a_used": None, "port_b_used": None,
                "matched": False, "score_a": 0.0, "score_b": 0.0, "method": None,
                "route_ports": None, "assumptions": assumptions, "error": "Invalid port inputs"}

    a0 = _norm_port(port_a)
    b0 = _norm_port(port_b)

    # direct
    d = _DISTANCE_MAP.get((a0, b0))
    if d is not None:
        return {"ok": True, "distance_nm": float(d), "port_a_used": a0, "port_b_used": b0,
                "matched": False, "score_a": 1.0, "score_b": 1.0, "method": "direct",
                "route_ports": [a0, b0], "assumptions": assumptions, "error": None}

    # route using table graph
    total, route = _shortest_path_distance(a0, b0, max_hops=int(max_hops))
    if total is not None:
        assumptions.append("Direct distance missing; used shortest path via intermediate ports in distance table.")
        return {"ok": True, "distance_nm": float(total), "port_a_used": a0, "port_b_used": b0,
                "matched": False, "score_a": 1.0, "score_b": 1.0, "method": "routed",
                "route_ports": route, "assumptions": assumptions, "error": None}

    # fuzzy port substitution then route
    a1, sa = _nearest_port(a0, cutoff=float(cutoff))
    b1, sb = _nearest_port(b0, cutoff=float(cutoff))
    if a1 is None or b1 is None:
        return {"ok": False, "distance_nm": None, "port_a_used": a1, "port_b_used": b1,
                "matched": True, "score_a": float(sa), "score_b": float(sb), "method": None,
                "route_ports": None, "assumptions": assumptions,
                "error": "No direct/route distance and could not match ports above cutoff"}

    assumptions.append(f"Ports not directly routable; substituted nearest ports by similarity: {a0}->{a1}, {b0}->{b1}.")

    total2, route2 = _shortest_path_distance(a1, b1, max_hops=int(max_hops))
    if total2 is None:
        return {"ok": False, "distance_nm": None, "port_a_used": a1, "port_b_used": b1,
                "matched": True, "score_a": float(sa), "score_b": float(sb), "method": "fuzzy_routed",
                "route_ports": None, "assumptions": assumptions,
                "error": f"No route found in distance table for matched ports {a1} <-> {b1}"}

    assumptions.append("Computed distance by summing known legs along shortest path in distance table.")
    return {"ok": True, "distance_nm": float(total2), "port_a_used": a1, "port_b_used": b1,
            "matched": True, "score_a": float(sa), "score_b": float(sb), "method": "fuzzy_routed",
            "route_ports": route2, "assumptions": assumptions, "error": None}



@tool
def get_speed_kn(vessel_name: str, vessel_load: str, speed_type: str) -> float | None:
    """
    Returns the speed of a given vessel in knots.
    Inputs:
      - vessel_name (unique)
      - vessel_load: 'BALLAST' or 'LADEN'
      - speed_type: 'ECON' or 'WARRANTED'
    Output: speed (float) or None if invalid/not found/missing
    """
    if not all(isinstance(x, str) and x.strip() for x in (vessel_name, vessel_load, speed_type)):
        return None

    vessel_load = vessel_load.strip().upper()
    speed_type = speed_type.strip().upper()

    col_map = {
        ("BALLAST", "ECON"): "econ_speed_ballast_kn",
        ("BALLAST", "WARRANTED"): "warranted_speed_ballast_kn",
        ("LADEN", "ECON"): "econ_speed_laden_kn",
        ("LADEN", "WARRANTED"): "warranted_speed_laden_kn",
    }
    col = col_map.get((vessel_load, speed_type))
    if col is None:
        return None

    key = _norm(vessel_name)
    if key not in _VESSEL_BY_NAME.index:
        return None

    val = _VESSEL_BY_NAME.loc[key].get(col, np.nan)
    return None if pd.isna(val) else float(val)


@tool
def get_dwt_mt(vessel_name: str) -> float | None:
    """
    Returns the deadweight tonnage (DWT) of a vessel in metric tonnes.
    Input: vessel_name
    Output: DWT (float) or None if not found/missing
    """
    if not isinstance(vessel_name, str) or not vessel_name.strip():
        return None

    key = _norm(vessel_name)
    if key not in _VESSEL_BY_NAME.index:
        return None

    val = _VESSEL_BY_NAME.loc[key].get("dwt_mt", np.nan)
    return None if pd.isna(val) else float(val)


@tool
def get_working_rate_mt_per_day(cargo_id: str, work_type: str) -> float | None:
    """
    Returns the working rate (mt/day) for a given cargo and work type.
    Inputs:
      - cargo_id (unique)
      - work_type: 'LOAD' or 'DISCHARGE'
    Output: rate (float) or None if invalid/not found/missing
    """
    if not all(isinstance(x, str) and x.strip() for x in (cargo_id, work_type)):
        return None

    work_type = work_type.strip().upper()
    col_map = {
        "LOAD": "load_rate_mt_per_day",
        "DISCHARGE": "discharge_rate_mt_per_day",
    }
    col = col_map.get(work_type)
    if col is None:
        return None

    key = _norm(cargo_id)
    if key not in _CARGO_BY_ID.index:
        return None

    val = _CARGO_BY_ID.loc[key].get(col, np.nan)
    return None if pd.isna(val) else float(val)


@tool
def get_cargo_quantity_mt(cargo_id: str) -> float | None:
    """
    Returns the cargo quantity in metric tonnes.
    Input: cargo_id (unique)
    Output: quantity (float) or None if not found/missing
    """
    if not isinstance(cargo_id, str) or not cargo_id.strip():
        return None

    key = _norm(cargo_id)
    if key not in _CARGO_BY_ID.index:
        return None

    val = _CARGO_BY_ID.loc[key].get("quantity_mt", np.nan)
    return None if pd.isna(val) else float(val)


@tool
def calculate_sailing_days(distance: float, speed: float) -> float | None:
    """
    Returns sailing days given distance (nm) and speed (kn).
    Inputs: distance (nm), speed (kn)
    Output: days (float) or None if invalid
    """
    if distance is None or speed is None:
        return None
    if not np.isfinite(distance) or not np.isfinite(speed) or speed <= 0:
        return None
    return float(distance) / float(speed) / 24.0


@tool
def calculate_port_working_days(cargo_qty_mt: float, working_rate_mt_per_day: float) -> float | None:
    """
    Returns port working days given cargo quantity (mt) and working rate (mt/day).
    Inputs: cargo_qty_mt, working_rate_mt_per_day
    Output: days (float) or None if invalid
    """
    if cargo_qty_mt is None or working_rate_mt_per_day is None:
        return None
    if not np.isfinite(cargo_qty_mt) or not np.isfinite(working_rate_mt_per_day):
        return None
    if cargo_qty_mt < 0 or working_rate_mt_per_day <= 0:
        return None
    return float(cargo_qty_mt) / float(working_rate_mt_per_day)


tools = [
    check_vessel_exists,
    check_cargo_exists,
    suggest_ports,
    get_distance_nm,
    get_speed_kn,
    get_dwt_mt,
    get_working_rate_mt_per_day,
    get_cargo_quantity_mt,
    calculate_sailing_days,
    calculate_port_working_days,
]


math_model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def math_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="\n".join(
            [
                "You are a dry bulk trader at Cargill Ocean Transportation Singapore, managing a fleet of Capesize vessels (large bulk carriers).",
                "You move bulk cargoes such as iron ore and bauxite across global trade routes for customers.",
                "You MUST use the provided tools for calculations and lookups from the CSV data (vessels, cargoes, port distances, etc.).",
                "When inputs are missing (e.g., vessel_load or speed_type), ask the user—do not assume.",
                "When a distance tool returns ok=True and method != 'direct', explicitly mention route_ports and assumptions in the reply.",
            ]
        )
    )

    response = math_model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    return "continue" if tool_calls else "end"


def run_chat(app):
    state: AgentState = {"messages": []}
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        state["messages"].append(HumanMessage(content=user_text))

        final_state = None
        for event in app.stream(state, stream_mode="values"):
            final_state = event
            msg = event["messages"][-1]

            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    print("\n🧠 Assistant wants to call tools:")
                    for tc in msg.tool_calls:
                        print(f"  🔧 {tc['name']}({tc['args']})")
                else:
                    print("\n🧠 Assistant:", msg.content)

            elif isinstance(msg, ToolMessage):
                print(f"\n📦 Tool result from `{msg.name}`:")
                print(f"  → {msg.content}")

        state = final_state


def main():
    graph = StateGraph(AgentState)
    graph.add_node("math_agent", math_agent)

    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "math_agent")
    graph.add_conditional_edges(
        "math_agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    graph.add_edge("tools", "math_agent")

    app = graph.compile()
    run_chat(app)


if __name__ == "__main__":
    main()
