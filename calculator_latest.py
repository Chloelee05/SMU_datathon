from typing import TypedDict
from datetime import date, timedelta

import z_data_in
import pandas as pd
import numpy as np
import heapq
import difflib



# -----------------------------
# define classes
# -----------------------------
class vessel(TypedDict):
    vessel_id: str
    vessel_name: str
    dwt_mt: float
    hire_rate_usd_per_day: float
    econ_speed_laden_kn: float
    econ_speed_ballast_kn: float
    warranted_speed_laden_kn: float
    warranted_speed_ballast_kn: float
    econ_fuel_laden_vlsf_mt_per_day: float
    econ_fuel_laden_mgo_mt_per_day: float
    econ_fuel_ballast_vlsf_mt_per_day: float
    econ_fuel_ballast_mgo_mt_per_day: float
    warranted_fuel_laden_vlsf_mt_per_day: float
    warranted_fuel_laden_mgo_mt_per_day: float
    warranted_fuel_ballast_vlsf_mt_per_day: float
    warranted_fuel_ballast_mgo_mt_per_day: float
    port_idle_mgo_mt_per_day: float
    port_working_mgo_mt_per_day: float
    current_port: str
    current_region: str
    etd_date: date
    rob_vlsf_mt: float
    rob_mgo_mt: float


class cargo(TypedDict):
    cargo_id: str
    start_region: str
    end_region: str
    customer: str
    commodity: str
    quantity_mt: float
    quantity_tolerance_pct: float
    quantity_type: str
    freight_rate_usd_per_mt: float
    commission_pct: float
    commission_type: str
    laycan_start_date: date
    laycan_end_date: date
    load_port: str
    load_port_remarks: str
    load_port_cost_usd: float
    load_rate_mt_per_day: float
    load_turn_time_hr: float
    load_type: str
    discharge_port: str
    discharge_port_remarks: str
    discharge_port_cost_usd: float
    discharge_rate_mt_per_day: float
    discharge_turn_time_hr: float
    discharge_type: str



# -----------------------------
# Joining (cargill vessel and market vessel) / (cargill cargo and market cargo)
# -----------------------------
def _norm(s: str) -> str:
    return str(s).strip().casefold()



# -----------------------------
# handling port location fits
# -----------------------------
def _norm_port(s: str) -> str:
    return str(s).strip().upper()


# globals initialized by init_port_data
ALIAS_TO_CANON: dict[str, str] = {}
DISTANCE_NM: dict[tuple[str, str], float] = {}
GRAPH: dict[str, list[tuple[str, float]]] = {}
VALID_PORTS: list[str] = []


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



def _nearest_port(port: str, *, cutoff: float = 0.85) -> str | None:
    p = _norm_port(port)
    if p in VALID_PORTS:
        return p
    matches = difflib.get_close_matches(p, VALID_PORTS, n=1, cutoff=float(cutoff))
    return matches[0] if matches else None



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




# -----------------------------
# handling FFA data
# -----------------------------
def ffa_month_col(d: date) -> str:
    # Feb_26, Mar_26, ...
    return f"{d.strftime('%b')}_{d.strftime('%y')}"

def ffa_quarter_col(d: date) -> str:
    # Q1_26, Q2_26, ...
    q = (d.month - 1) // 3 + 1
    return f"Q{q}_{d.strftime('%y')}"

def ffa_calendar_col(d: date) -> str:
    # Cal_26, Cal_27, ...
    return f"Cal_{d.strftime('%y')}"

def normalize_route_key(route_key: str) -> str:
    # accept "C3", "c3", "C3 (Tubarao-Qingdao)" etc
    s = str(route_key).strip().upper()
    # keep only first token for C3/C5/C7/5TC style
    return s.split()[0]  # "C3", "C5", "C7", "5TC"

def ffa_route_row(ffa: pd.DataFrame, route_key: str) -> pd.Series:
    key = normalize_route_key(route_key)

    # exact match first (e.g., "5TC")
    m = ffa["Route"].astype(str).str.upper().eq(key)
    if m.any():
        return ffa.loc[m].iloc[0]

    # prefix match for "C3 (..)" etc
    m = ffa["Route"].astype(str).str.upper().str.startswith(key)
    if m.any():
        return ffa.loc[m].iloc[0]

    raise KeyError(f"Route '{route_key}' not found in FFA table")

def get_ffa_rate_usd_per_day(
    ffa: pd.DataFrame,
    route_key: str,
    d: date,
    *,
    fallback_order: tuple[str, ...] = ("month", "quarter", "cal"),
) -> float:
    """
    Returns FFA rate (USD/day) for given route + date.
    fallback_order decides how you map date->tenor columns:
      ("month","quarter","cal") is typical.
    """
    row = ffa_route_row(ffa, route_key)

    col_map = {
        "month": ffa_month_col(d),
        "quarter": ffa_quarter_col(d),
        "cal": ffa_calendar_col(d),
    }

    for key in fallback_order:
        col = col_map[key]
        if col in row.index and pd.notna(row[col]):
            return float(row[col])

    tried = [col_map[k] for k in fallback_order]
    raise KeyError(f"No FFA tenor available for route={route_key}, date={d}, tried={tried}")

def pick_ffa_route_from_regions(start_region: str, end_region: str) -> str:
    s = str(start_region).strip().upper()
    e = str(end_region).strip().upper()

    # C3: Brazil -> China
    if s == "BRAZIL" and e == "CHINA":
        return "C3"

    # C5: Australia -> China
    if s == "AUSTRALIA" and e == "CHINA":
        return "C5"

    # C7: Africa -> Europe
    if e == "EUROPE" and s in {"WEST_AFRICA", "SOUTH_AFRICA"}:
        return "C7"

    # fallback: generic capesize market hire
    return "5TC"


# -----------------------------
# handling fuel data
# -----------------------------
def pick_bunker_location_for_load_port(
    bunker: pd.DataFrame,
    load_port: str,
    *,
    fallback_location: str = "Singapore",
    max_hops: int = 6,
) -> str:
    """
    Choose the closest bunker pricing location to load_port.
    - If load_port itself is a bunker location, use it.
    - Else, compute shortest path distance (nm) from load_port to each bunker Location
      using your port graph (get_distance_nm_safe).
    - If no distances are available to any bunker location, fall back to fallback_location.
    """
    lp = canon_port(load_port)

    bunker_locs = (
        bunker["Location"]
        .astype(str)
        .map(_norm_port)
        .dropna()
        .unique()
        .tolist()
    )

    # Exact match: if load port is directly a bunker location
    if _norm_port(lp) in bunker_locs:
        return lp

    best_loc = None
    best_dist = float("inf")

    for loc in bunker_locs:
        d = get_distance_nm_safe(lp, loc, max_hops=max_hops)
        if d is None:
            continue
        if d < best_dist:
            best_dist = d
            best_loc = loc

    return best_loc if best_loc is not None else fallback_location


def bunker_price_usd_per_mt_by_load_port(
    bunker: pd.DataFrame,
    load_port: str,
    fuel: str,
    d: date,
    *,
    fallback_location: str = "Singapore",
    max_hops: int = 6,
) -> tuple[str, float]:
    """
    Returns (chosen_location, price_usd_per_mt) where chosen_location is
    the closest bunker Location to the load_port (or fallback).
    """
    loc = pick_bunker_location_for_load_port(
        bunker,
        load_port,
        fallback_location=fallback_location,
        max_hops=max_hops,
    )
    price = bunker_price_usd_per_mt(bunker, loc, fuel, d)
    return loc, price


def bunker_price_usd_per_mt(bunker: pd.DataFrame, location: str, fuel: str, d: date) -> float:
    col = f"{d.strftime('%b')}_{d.strftime('%y')}"  # Feb_26 etc
    row = bunker[(bunker["Location"].str.upper() == location.upper()) & (bunker["Fuel"].str.upper() == fuel.upper())]
    if row.empty or col not in row.columns or pd.isna(row.iloc[0][col]):
        raise KeyError(f"Missing bunker price for {location=} {fuel=} {col=}")
    return float(row.iloc[0][col])



# -----------------------------
# helper functions
# -----------------------------

def cargo_min_max_mt(qty_mt: float, tol_pct: float) -> tuple[float, float]:
    """
    Returns a list of min and max values for cargo quantity_mt
    """
    tol = tol_pct / 100.0
    return qty_mt * (1 - tol), qty_mt * (1 + tol)



def capacity_feasible_moloo(dwt_mt: float, qty_mt: float, tol_pct: float) -> bool:
    """
    Returns a bool depending on whether the ship can carry the min quantity
    """
    q_min, _ = cargo_min_max_mt(qty_mt, tol_pct)
    return dwt_mt >= q_min



def get_sailing_days(distance_nm: float, speed_kn: float, SPEED_MULTIPLIER: float) -> float:
    """
    Returns the number of sailing days given distance and speed
    """
    if speed_kn <= 0:
        raise ValueError(f"Invalid speed_kn={speed_kn}")
    
    if distance_nm < 0:
        raise ValueError(f"Invalid distance_nm={distance_nm}")
    
    if SPEED_MULTIPLIER <= 0:
        raise ValueError(f"Invalid SPEED_MULTIPLIER={SPEED_MULTIPLIER}")
    
    return distance_nm / (speed_kn * SPEED_MULTIPLIER) / 24



def prune_eta_to_load_port(etd_date: date, laycan_end_date: date, sailing_days: float) -> bool:
    """
    Returns True if the vessel can reach the load port on or before laycan_end_date.
    Returns False otherwise.
    """
    eta_load = etd_date + timedelta(days=sailing_days)
    return eta_load <= laycan_end_date



def calculate(PRUNE: bool, SPEED: str, DWT_MULTIPLIER: float, SPEED_MULTIPLIER: float,LOAD_PORT_DELAY: float, DISCHARGE_PORT_DELAY: float, VLSF_BUFFER_PCT: float, MGO_BUFFER_PCT: float,BUNKER_PORT_COST: float) -> None:
    # -----------------------------
    # get list of VESSEL_IDS and CARGO_IDS
    # -----------------------------
    VESSEL_IDS = _ALL_VESSELS["vessel_id"].astype(str).dropna().unique().tolist()
    CARGO_IDS  = _ALL_CARGOES["cargo_id"].astype(str).dropna().unique().tolist()


    # -----------------------------
    # get speed and fuel consumption
    # -----------------------------
    if SPEED == "ECON":
        speed_ballast_str = "econ_speed_ballast_kn"
        speed_laden_str = "econ_speed_laden_kn"
        fuel_ballast_vlsf_mt_per_day = "econ_fuel_ballast_vlsf_mt_per_day"
        fuel_ballast_mgo_mt_per_day = "econ_fuel_ballast_mgo_mt_per_day"
        fuel_laden_vlsf_mt_per_day = "econ_fuel_laden_vlsf_mt_per_day"
        fuel_laden_mgo_mt_per_day = "econ_fuel_laden_mgo_mt_per_day"

    elif SPEED == "WARRANTED":
        speed_ballast_str = "warranted_speed_ballast_kn"
        speed_laden_str = "warranted_speed_laden_kn"
        fuel_ballast_vlsf_mt_per_day = "warranted_fuel_ballast_vlsf_mt_per_day"
        fuel_ballast_mgo_mt_per_day = "warranted_fuel_ballast_mgo_mt_per_day"
        fuel_laden_vlsf_mt_per_day = "warranted_fuel_laden_vlsf_mt_per_day"
        fuel_laden_mgo_mt_per_day = "warranted_fuel_laden_mgo_mt_per_day"
    
    else:
        raise ValueError(f"Invalid SPEED='{SPEED}', must be 'ECON' or 'WARRANTED'")


    # -----------------------------
    # create indexed df to improve performance
    # -----------------------------
    _VESSEL_BY_ID = _ALL_VESSELS.set_index("vessel_id", drop=False)
    _CARGO_BY_ID  = _ALL_CARGOES.set_index("cargo_id", drop=False)



    # -----------------------------
    # defining variables
    # -----------------------------
    missing_ballast_dist = 0
    missing_laden_dist = 0
    prune_1 = 0
    prune_2 = 0
    rows = []

    # validate cargill freight rates (all offending ids)
    missing_mask = (_ALL_CARGOES["_src"] == "cargill") & (_ALL_CARGOES["freight_rate_usd_per_mt"].isna())
    if missing_mask.any():
        bad_ids = _ALL_CARGOES.loc[missing_mask, "cargo_id"].astype(str).dropna().tolist()
        raise ValueError(f"Missing cargill freight rate for cargo_id={bad_ids}")

    # validate dates (all offending ids)
    if "etd_date" in _ALL_VESSELS.columns:
        bad_mask = _ALL_VESSELS["etd_date"].isna()
        if bad_mask.any():
            bad_ids = _ALL_VESSELS.loc[bad_mask, "vessel_id"].astype(str).dropna().tolist()
            raise ValueError(f"Missing vessel etd_date for vessel_id={bad_ids}")

    if "laycan_start_date" in _ALL_CARGOES.columns or "laycan_end_date" in _ALL_CARGOES.columns:
        cols = [c for c in ["laycan_start_date", "laycan_end_date"] if c in _ALL_CARGOES.columns]
        bad_mask = _ALL_CARGOES[cols].isna().any(axis=1)
        if bad_mask.any():
            bad_ids = _ALL_CARGOES.loc[bad_mask, "cargo_id"].astype(str).dropna().tolist()
            raise ValueError(f"Missing cargo laycan dates for cargo_id={bad_ids}")



    # -----------------------------
    # loop for each vessel - cargo pair
    # -----------------------------
    for vid in VESSEL_IDS:
        v = _VESSEL_BY_ID.loc[vid]

        for cid in CARGO_IDS:
            c = _CARGO_BY_ID.loc[cid]

            # calculate dwt_mt available
            effective_capacity = v["dwt_mt"] * DWT_MULTIPLIER

            # enables or disables pruning
            if PRUNE:
                # pruning case #1 - Vessel cant load min quantity of cargo
                if not capacity_feasible_moloo(effective_capacity, c["quantity_mt"], c["quantity_tolerance_pct"]):
                    prune_1 += 1
                    continue

            # get ballast distance between vessel's current_port and cargo's load_port
            ballast_dist = get_distance_nm_safe(v["current_port"], c["load_port"])

            # prints error and skips if no distance available (should not happen after integration of alt ports)========================================(remove print in prod)
            if ballast_dist is None:
                missing_ballast_dist += 1
                print(f"ERROR: Unable to get distance between {v['current_port']} and {c['load_port']}")
                continue

            # get ballast days spent sailing
            ballast_days = get_sailing_days(ballast_dist, v[speed_ballast_str], SPEED_MULTIPLIER)

            # enables or disables pruning
            if PRUNE: 
                # pruning case #2 - Reach load_port before laycan_end_date
                if not prune_eta_to_load_port(
                    v["etd_date"],
                    c["laycan_end_date"],
                    ballast_days
                ):
                    prune_2 += 1
                    continue
            
            # get laden distance between cargo's load_port and discharge port
            laden_dist   = get_distance_nm_safe(c["load_port"], c["discharge_port"])

            # prints error and skips if no distance available (should not happen after integration of alt ports)========================================(remove print in prod)
            if laden_dist is None:
                missing_laden_dist += 1
                print(f"ERROR: Unable to get distance between {c['load_port']} and {c['discharge_port']}")
                continue
                
            # get laden days spent sailing
            laden_days = get_sailing_days(laden_dist, v[speed_laden_str], SPEED_MULTIPLIER)

            # get loaded_qty_mt
            q_min, q_max = cargo_min_max_mt(c["quantity_mt"], c["quantity_tolerance_pct"])
            loaded_qty_mt = min(q_max, effective_capacity)   

            # calculate eta at loading port
            eta_load = v["etd_date"] + timedelta(days=ballast_days)

            # calculate waiting time before laycan start date
            delta = c["laycan_start_date"] - eta_load
            load_port_wait_days = max(0.0, delta.total_seconds() / 86400.0)


            # calculate loading and discharge times
            load_rate = float(c["load_rate_mt_per_day"])
            disc_rate = float(c["discharge_rate_mt_per_day"])

            if load_rate <= 0 or disc_rate <= 0:
                continue    # bad input row; skip

            load_days = loaded_qty_mt / load_rate
            discharge_days = loaded_qty_mt / disc_rate


            # calculate total port idle days
            total_port_idle_days = (
                load_port_wait_days
                + (c["load_turn_time_hr"] / 24)
                + (c["discharge_turn_time_hr"] / 24)
                + float(LOAD_PORT_DELAY)
                + float(DISCHARGE_PORT_DELAY)
            )

            # calculate total port working days
            total_port_working_days = load_days + discharge_days



            # calculate total days
            total_days = ballast_days + laden_days + total_port_idle_days + total_port_working_days

            if v["_src"] == "cargill":
                hire_rate = float(v["hire_rate_usd_per_day"])
            else:
                route_key = pick_ffa_route_from_regions(c["start_region"], c["end_region"])
                hire_rate = get_ffa_rate_usd_per_day(ffa, route_key, c["laycan_start_date"])

            # calculating total_hire_cost
            total_hire_cost = total_days * hire_rate

            # calculating port costs
            port_costs = float(c["load_port_cost_usd"]) + float(c["discharge_port_cost_usd"])

            # get freight rate or infer (market cargoes have no freight rate)
            if pd.isna(c["freight_rate_usd_per_mt"]):
                if c["_src"] == "market":
                    freight_rate = (total_hire_cost + port_costs) / loaded_qty_mt
                    inferred = True
                else:
                    raise ValueError(f"Missing cargill freight rate for cargo_id={c['cargo_id']}")
            else:
                freight_rate = float(c["freight_rate_usd_per_mt"])
                inferred = False

            # calculating revenue
            gross_revenue = loaded_qty_mt * freight_rate

            # calculating commission
            commission_pct = float(c["commission_pct"]) if pd.notna(c["commission_pct"]) else 0.0
            commission = gross_revenue * (commission_pct / 100.0)

            # calculating net revenue (-commission)
            net_revenue = gross_revenue - commission

            # calculalting profits pre-bunker
            profit_pre_bunker = net_revenue - total_hire_cost - port_costs



            # sailing fuel
            ballast_vlsf = ballast_days * float(v[fuel_ballast_vlsf_mt_per_day])
            ballast_mgo  = ballast_days * float(v[fuel_ballast_mgo_mt_per_day])
            laden_vlsf   = laden_days   * float(v[fuel_laden_vlsf_mt_per_day])
            laden_mgo    = laden_days   * float(v[fuel_laden_mgo_mt_per_day])

            # port fuel (idle + working)
            port_idle_mgo    = total_port_idle_days    * float(v["port_idle_mgo_mt_per_day"])
            port_working_mgo = total_port_working_days * float(v["port_working_mgo_mt_per_day"])

            total_vlsf = (ballast_vlsf + laden_vlsf) / 100 * (100 + (VLSF_BUFFER_PCT * 100))
            total_mgo  = (ballast_mgo + laden_mgo + port_idle_mgo + port_working_mgo) / 100 * (100 + (MGO_BUFFER_PCT * 100))

            total_vlsf = max(0, total_vlsf - v["rob_vlsf_mt"])
            total_mgo  = max(0, total_mgo  - v["rob_mgo_mt"])


            # fuel price by closest bunker location to load port
            bunker_loc_vlsf, p_vlsf = bunker_price_usd_per_mt_by_load_port(
                bunker,
                c["load_port"],
                "VLSFO",
                eta_load,
                fallback_location="Singapore",
            )

            bunker_loc_mgo, p_mgo = bunker_price_usd_per_mt_by_load_port(
                bunker,
                c["load_port"],
                "MGO",
                eta_load,
                fallback_location="Singapore",
            )

            bunker_loc = bunker_loc_vlsf

            # calculate bunker cost
            bunker_cost = (total_vlsf * p_vlsf) + (total_mgo * p_mgo) + BUNKER_PORT_COST

            # calculate profit post bunker
            profit_post_bunker = profit_pre_bunker - bunker_cost

            # TCE (implied time charter equivalent, excludes hire)
            tce_usd_per_day = (net_revenue - port_costs - bunker_cost) / total_days

            # calculate profit per day
            profit_usd_per_day = profit_post_bunker / total_days




            # Contribution (hire cancels out for owned/Cargill vessels vs being idle)
            contribution_usd = net_revenue - port_costs - bunker_cost
            contribution_usd_per_day = contribution_usd / total_days

            # Decision metric:
            # - Cargill vessel: maximize contribution (since hire is sunk anyway)
            # - Market vessel: maximize profit post bunker (since hire is avoidable)
            decision_profit = contribution_usd if v["_src"] == "cargill" else profit_post_bunker
            decision_profit_per_day = decision_profit / total_days





            rows.append({
                "vessel_id": v["vessel_id"],
                "cargo_id": c["cargo_id"],
                "src_vessel": v["_src"],
                "hire_rate_usd_per_day": hire_rate,

                "ballast_nm": ballast_dist,
                "laden_nm": laden_dist,
                "ballast_days": ballast_days,
                "laden_days": laden_days,

                "eta_load": eta_load,
                "wait_days": load_port_wait_days,
                "port_idle_days": total_port_idle_days,
                "port_work_days": total_port_working_days,
                "total_days": total_days,

                "loaded_qty_mt": loaded_qty_mt,

                "gross_revenue_usd": gross_revenue,
                "freight_inferred": inferred,
                "commission_usd": commission,
                "net_revenue_usd": net_revenue,
                "port_cost_usd": port_costs,
                "hire_cost_usd": total_hire_cost,
                "profit_pre_bunker_usd": profit_pre_bunker,

                "total_vlsf": total_vlsf,
                "total_mgo": total_mgo,
                "bunker_location": bunker_loc,
                "bunker_price_vlsf": p_vlsf,
                "bunker_price_mgo": p_mgo,
                "bunker_cost": bunker_cost,

                "profit_post_bunker": profit_post_bunker,
                "contribution_usd": contribution_usd,
                "contribution_usd_per_day": contribution_usd_per_day,
                "decision_profit": decision_profit,
                "decision_profit_per_day": decision_profit_per_day,
                "tce_usd_per_day": tce_usd_per_day,
                "profit_usd_per_day": profit_usd_per_day
            })
    
    print("Missing ballast dist: ", missing_ballast_dist)
    print("Missing laden_dist", missing_laden_dist)
    print("Prune 1: ", prune_1)
    print("Prune 2", prune_2)

    df = pd.DataFrame(rows)

    # build lookups
    cargo_src = _ALL_CARGOES.set_index("cargo_id")["_src"]
    vessel_name_map = _ALL_VESSELS.set_index("vessel_id")["vessel_name"]

    df["src_cargo"] = df["cargo_id"].map(cargo_src)
    df["vessel_name"] = df["vessel_id"].map(vessel_name_map)

    df_cargill = df[df["src_cargo"] == "cargill"]

    best_per_cargo = (
        df_cargill
        .sort_values("decision_profit_per_day", ascending=False)
        .groupby("cargo_id", as_index=False)
        .first()
    )

    # save to csv
    df.to_csv("base_cases.csv", index=False)

    # print
    cols = [
        "cargo_id",
        "vessel_id",
        "vessel_name",
        "src_vessel",
        "decision_profit_per_day",
        "contribution_usd_per_day",
        "tce_usd_per_day",
        "profit_usd_per_day",
        "hire_rate_usd_per_day",
        "total_days",
        "bunker_location",
    ]

    print(best_per_cargo[cols].to_string(index=False))







if __name__ == "__main__":

    # define constants
    PRUNE = True                # True / False              --> for enabling / disabling pruning
    SPEED = "ECON"              # "ECON" / "WARRANTED"
    DWT_MULTIPLIER = 1          # 0 < DWT_MULTIPLIER < 1    --> for setting buffer in scenario
    SPEED_MULTIPLIER = 1        # 0 < SPEED_MULTIPLIER < 1  --> for setting in scenario (slow down due to weather, etc)
    LOAD_PORT_DELAY = 0         # > 0                       --> for scenario
    DISCHARGE_PORT_DELAY = 0    # > 0                       --> for scenario
    VLSF_BUFFER_PCT = 0         # 0 < VLSF_BUFFER < 1       --> for setting buffer in scenario 
    MGO_BUFFER_PCT = 0          # 0 < MGO_BUFFER < 1       --> for setting buffer in scenario 
    BUNKER_PORT_COST = 5000

    PORT_ALIASES = {
        "KAMSAR ANCHORAGE": [
            "KAMSAR",
            "PORT KAMSAR",
        ],
        "MAP TA PHUT": ["MAPTAPHUT"],
        "GWANGYANG LNG TERMINAL": ["GWANGYANG"],
        "VANCOUVER (CANADA)": ["VANCOUVER"]
    }
    
    # -----------------------------
    # Load data
    # -----------------------------
    cargill_vessels, cargill_cargoes, market_vessels, market_cargoes, port_distances, ffa, bunker = z_data_in.load_data()


    # -----------------------------
    # Joining (cargill vessel and market vessel) / (cargill cargo and market cargo)
    # -----------------------------
    # Vessels: unique vessel_name across combined set and cache for faster lookup
    _ALL_VESSELS = pd.concat(
        [cargill_vessels.assign(_src="cargill"), market_vessels.assign(_src="market")],
        ignore_index=True,
    ).copy()
    _ALL_VESSELS["_name_key"] = _ALL_VESSELS["vessel_name"].astype(str).map(_norm)
    _VESSEL_BY_NAME = _ALL_VESSELS.set_index("_name_key", drop=False)


    # Cargoes: unique cargo_id across combined set and cache for faster lookup
    _ALL_CARGOES = pd.concat(
        [cargill_cargoes.assign(_src="cargill"), market_cargoes.assign(_src="market")],
        ignore_index=True,
    ).copy()
    _ALL_CARGOES["_id_key"] = _ALL_CARGOES["cargo_id"].astype(str).map(_norm)
    _CARGO_BY_ID = _ALL_CARGOES.set_index("_id_key", drop=False)


    # enforce uniqueness
    dupe_v = _ALL_VESSELS["_name_key"].duplicated(keep=False)
    if dupe_v.any():
        dupes = _ALL_VESSELS.loc[dupe_v, ["vessel_name", "_src"]].sort_values("vessel_name")
        raise ValueError(f"Duplicate vessel_name keys found:\n{dupes.to_string(index=False)}")

    dupe_c = _ALL_CARGOES["_id_key"].duplicated(keep=False)
    if dupe_c.any():
        dupes = _ALL_CARGOES.loc[dupe_c, ["cargo_id", "_src"]].sort_values("cargo_id")
        raise ValueError(f"Duplicate cargo_id keys found:\n{dupes.to_string(index=False)}")


    init_port_data(port_distances, PORT_ALIASES)





    calculate(PRUNE, SPEED, DWT_MULTIPLIER, SPEED_MULTIPLIER, LOAD_PORT_DELAY, DISCHARGE_PORT_DELAY, VLSF_BUFFER_PCT, MGO_BUFFER_PCT, BUNKER_PORT_COST)








