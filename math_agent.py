from typing import Annotated, TypedDict
import difflib, heapq

import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import z_data_in


# =========================
# Load data once
# =========================
cargill_vessels, cargill_cargoes, market_vessels, market_cargoes, port_distances, ffa, bunker = (
    z_data_in.load_data()
)


# =========================
# Agent state
# =========================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =========================
# Helpers / caches
# =========================
def _norm(s: str) -> str:
    return str(s).strip().casefold()


def _norm_port(s: str) -> str:
    return str(s).strip().upper()


# =========================
# Port distance logic (same as your get_distance_nm_safe)
# =========================
# globals initialized by init_port_data
ALIAS_TO_CANON: dict[str, str] = {}
DISTANCE_NM: dict[tuple[str, str], float] = {}
GRAPH: dict[str, list[tuple[str, float]]] = {}
VALID_PORTS: list[str] = []  # only used by _nearest_port


def canon_port(p: str) -> str:
    x = _norm_port(p)
    return ALIAS_TO_CANON.get(x, x)


def _shortest_path_nm(start: str, goal: str, *, max_hops: int = 6) -> float | None:
    start = _norm_port(start)
    goal = _norm_port(goal)
    if start == goal:
        return 0.0

    # Dijkstra with hop limit
    dist: dict[str, float] = {start: 0.0}
    hops: dict[str, int] = {start: 0}
    pq: list[tuple[float, str]] = [(0.0, start)]
    visited: set[str] = set()

    while pq:
        dcur, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return dcur

        for nxt, w in GRAPH.get(node, []):
            nh = hops[node] + 1
            if nh > max_hops:
                continue
            nd = dcur + w
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                hops[nxt] = nh
                heapq.heappush(pq, (nd, nxt))

    return None



def get_distance_nm_safe(port_a: str, port_b: str, *, max_hops: int = 6) -> float | None:
    a = canon_port(port_a)
    b = canon_port(port_b)
    if a == b:
        return 0.0
    d = DISTANCE_NM.get((a, b))
    if d is not None:
        return d
    return _shortest_path_nm(a, b, max_hops=max_hops)


def init_port_data(port_distances: pd.DataFrame, port_aliases: dict[str, list[str] | str]) -> None:
    global ALIAS_TO_CANON, DISTANCE_NM, GRAPH, VALID_PORTS

    # flatten to alias -> canonical
    ALIAS_TO_CANON = {}
    for canon, aliases in port_aliases.items():
        canon_n = _norm_port(canon)
        if isinstance(aliases, str):
            aliases = [aliases]
        for a in aliases:
            a_n = _norm_port(a)
            ALIAS_TO_CANON[a_n] = canon_n

    # Build distance lookup + adjacency graph
    _pd = port_distances.copy()
    _pd["A"] = _pd["PORT_NAME_FROM"].astype(str).map(_norm_port)
    _pd["B"] = _pd["PORT_NAME_TO"].astype(str).map(_norm_port)
    _pd["D"] = pd.to_numeric(_pd["DISTANCE"], errors="coerce")

    DISTANCE_NM = {}
    GRAPH = {}
    for a, b, d in zip(_pd["A"], _pd["B"], _pd["D"]):
        if pd.notna(d):
            dd = float(d)
            DISTANCE_NM[(a, b)] = dd
            DISTANCE_NM[(b, a)] = dd
            GRAPH.setdefault(a, []).append((b, dd))
            GRAPH.setdefault(b, []).append((a, dd))

    VALID_PORTS = sorted({p for (x, y) in DISTANCE_NM.keys() for p in (x, y)})


# =========================
# Build vessel/cargo lookup tables
# =========================
_ALL_VESSELS = pd.concat(
    [cargill_vessels.assign(_src="cargill"), market_vessels.assign(_src="market")],
    ignore_index=True,
)
_ALL_VESSELS["_key"] = _ALL_VESSELS["vessel_name"].map(_norm)
_VESSEL_BY_NAME = _ALL_VESSELS.set_index("_key")

_ALL_CARGOES = pd.concat(
    [cargill_cargoes.assign(_src="cargill"), market_cargoes.assign(_src="market")],
    ignore_index=True,
)
_ALL_CARGOES["_key"] = _ALL_CARGOES["cargo_id"].map(_norm)
_CARGO_BY_ID = _ALL_CARGOES.set_index("_key")


# =========================
# Initialize port data (aliases + graph)
# =========================
PORT_ALIASES = {
    "KAMSAR ANCHORAGE": [
        "KAMSAR",
        "PORT KAMSAR",
    ],
    "MAP TA PHUT": ["MAPTAPHUT"],
    "GWANGYANG LNG TERMINAL": ["GWANGYANG"],
    "VANCOUVER (CANADA)": ["VANCOUVER"],
}
init_port_data(port_distances, PORT_ALIASES)


# =========================
# Tools
# =========================
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
def list_cargill_vessels() -> dict:
    """
    List all Cargill vessels (cv_*).
    Output:
      {
        "ok": bool,
        "vessels": [{"vessel_id": str, "vessel_name": str}],
        "count": int
      }
    """
    df = _ALL_VESSELS[_ALL_VESSELS["_src"] == "cargill"]

    vessels = [
        {
            "vessel_id": row["vessel_id"],
            "vessel_name": row["vessel_name"],
        }
        for _, row in df.iterrows()
    ]

    return {
        "ok": True,
        "vessels": vessels,
        "count": len(vessels),
    }


@tool
def list_cargill_cargoes(include_summary: bool = True, limit: int = 50) -> dict:
    """
    List Cargill cargoes.
    If include_summary=True, includes route + laycan for context (no economics).
    """
    df = _ALL_CARGOES[_ALL_CARGOES["_src"] == "cargill"].copy()

    try:
        lim = int(limit)
    except Exception:
        lim = 50
    lim = max(1, min(200, lim))

    rows = []
    for _, row in df.head(lim).iterrows():
        item = {"cargo_id": row.get("cargo_id")}

        if include_summary:
            load_port = row.get("load_port")
            dis_port = row.get("discharge_port")

            item["route"] = f"{load_port} → {dis_port}"
            item["laycan_start_date"] = row.get("laycan_start_date")
            item["laycan_end_date"] = row.get("laycan_end_date")

            # ✅ PUT THIS LINE HERE
            item["label"] = (
                f"{item['cargo_id']} | "
                f"{load_port} → {dis_port} | "
                f"{item['laycan_start_date']}-{item['laycan_end_date']}"
            )

        rows.append(item)

    return {
        "ok": True,
        "cargoes": rows,
        "count": int(len(df)),
        "returned": len(rows),
    }



@tool
def list_market_vessels() -> dict:
    """
    List all Market vessels (mv_*).

    Output:
      {
        "ok": bool,
        "vessels": [{"vessel_id": str, "vessel_name": str}],
        "count": int
      }
    """
    df = _ALL_VESSELS[_ALL_VESSELS["_src"] == "market"]

    vessels = [
        {
            "vessel_id": row["vessel_id"],
            "vessel_name": row["vessel_name"],
        }
        for _, row in df.iterrows()
    ]

    return {
        "ok": True,
        "vessels": vessels,
        "count": len(vessels),
    }


@tool
def list_market_cargoes(include_summary: bool = True, limit: int = 50) -> dict:
    """
    List Market cargoes.
    If include_summary=True, includes route + laycan for context (no economics).
    """
    df = _ALL_CARGOES[_ALL_CARGOES["_src"] == "market"].copy()

    try:
        lim = int(limit)
    except Exception:
        lim = 50
    lim = max(1, min(200, lim))

    rows = []
    for _, row in df.head(lim).iterrows():
        item = {"cargo_id": row.get("cargo_id")}

        if include_summary:
            load_port = row.get("load_port")
            dis_port = row.get("discharge_port")

            item["route"] = f"{load_port} → {dis_port}"
            item["laycan_start_date"] = row.get("laycan_start_date")
            item["laycan_end_date"] = row.get("laycan_end_date")

            # ✅ PUT THIS LINE HERE
            item["label"] = (
                f"{item['cargo_id']} | "
                f"{load_port} → {dis_port} | "
                f"{item['laycan_start_date']}-{item['laycan_end_date']}"
            )

        rows.append(item)

    return {
        "ok": True,
        "cargoes": rows,
        "count": int(len(df)),
        "returned": len(rows),
    }


@tool
def get_distance_nm(port_a: str, port_b: str, max_hops: int = 6) -> float | None:
    """
    Returns the distance between 2 ports. Uses data only from the distance table provided. If direct distance missing, make assumptions:
      - route via intermediate ports (shortest path over table)
      - nearest port substitution if port not found
      - Never invents distances outside the table; only sums known legs.
      - Always state what assumptions are taken if any.
    """
    return get_distance_nm_safe(port_a, port_b, max_hops=max_hops)


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


math_tools = [
    check_vessel_exists,
    check_cargo_exists,

    # listing tools
    list_cargill_vessels,
    list_market_vessels,
    list_cargill_cargoes,
    list_market_cargoes,

    get_distance_nm,
    get_speed_kn,
    get_dwt_mt,
    get_working_rate_mt_per_day,
    get_cargo_quantity_mt,
    calculate_sailing_days,
    calculate_port_working_days,
]

math_tool_node = ToolNode(tools=math_tools)

math_model = ChatOpenAI(model="gpt-4o").bind_tools(math_tools)


def math_agent(state: AgentState) -> AgentState:
    system = SystemMessage(
        content=(
            "You are a math agent working as a Cargill dry bulk trader.\n"
            "Use tools for ALL calculations and lookups.\n"
            "If data is missing, ask the user."
        )
    )
    response = math_model.invoke([system] + state["messages"])
    return {"messages": [response]}
