from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import data_in


@dataclass
class Scenario:
    bunker_location: str = "Durban"
    bunker_month_col: str = "Apr_26"
    load_wait_days: float = 0.0
    discharge_wait_days: float = 0.0
    include_ballast: bool = False
    china_port_delay_days: float = 0.0
    bunker_price_uplift_pct: float = 0.0


def to_float_or_default(x, default: float = 0.0) -> float:
    if x is None:
        return default
    s = str(x).strip()
    if s in {"", "-", "—", "–"}:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "")
    return df


def normalize_series_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index.astype(str).str.strip().str.replace("\ufeff", "")
    return s


PORT_ALIASES = {
    "GWANGYANG": "KWANGYANG",
}


def _norm_port(x: Any) -> str:
    p = str(x).strip().upper()
    return PORT_ALIASES.get(p, p)


def get_distance_nm(port_distances: pd.DataFrame, a: str, b: str) -> float:
    a = _norm_port(a)
    b = _norm_port(b)

    q = port_distances.query("PORT_NAME_FROM == @a and PORT_NAME_TO == @b")["DISTANCE"]
    if len(q) == 0:
        q = port_distances.query("PORT_NAME_FROM == @b and PORT_NAME_TO == @a")["DISTANCE"]
    if len(q) == 0:
        raise ValueError(f"Distance not found: {a} <-> {b}")
    return float(q.iloc[0])


def get_bunker_price(
    bunker: pd.DataFrame,
    location: str,
    fuel: str,
    month_col: str,
    uplift_pct: float = 0.0,
) -> float:
    q = bunker.query("Location == @location and Fuel == @fuel")[month_col]
    if len(q) == 0:
        raise ValueError(f"Bunker price not found: {location}, {fuel}, {month_col}")
    base = float(q.iloc[0])
    return base * (1.0 + uplift_pct / 100.0)


CHINA_PORT_HINTS = {
    "QINGDAO", "DAO", "CNSHA", "SHANGHAI", "TIANJIN", "DALIAN", "NINGBO", "GUANGZHOU",
    "RIZHAO", "CAOF", "CAOFEIDIAN", "LIANYUNGANG", "ZHOUSHAN", "XINGANG",
}


def is_china_port(port_name: str) -> bool:
    p = _norm_port(port_name)
    if p in CHINA_PORT_HINTS:
        return True
    return any(h in p for h in ["CHINA", "CNSHA", "CN", "QINGDAO", "SHANGHAI"])


def calc_voyage(
    vessel: pd.Series,
    cargo: pd.Series,
    port_distances: pd.DataFrame,
    bunker: pd.DataFrame,
    scenario: Scenario,
    *,
    risk_model=None,
    default_hire_usd_per_day: Optional[float] = None,
) -> Dict[str, Any]:
    vessel = normalize_series_index(vessel)
    cargo = normalize_series_index(cargo)

    load_port = _norm_port(cargo.get("load_port"))
    discharge_port = _norm_port(cargo.get("discharge_port"))

    dist_laden = get_distance_nm(port_distances, load_port, discharge_port)

    dist_ballast = 0.0
    ballast_days = 0.0

    ml_risk = None
    weather_factor = 1.0
    port_delay_days = 0.0
    load_port_delay_days = 0.0
    discharge_port_delay_days = 0.0

    if risk_model is not None:
        try:
            month_map = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            }
            month = 1
            if isinstance(scenario.bunker_month_col, str):
                token = scenario.bunker_month_col.strip().split("_")[0][:3].upper()
                month = month_map.get(token, 1)
            ml_out = risk_model.predict({
                "from_port": load_port,
                "to_port": discharge_port,
                "month": month,
                "distance_nm": dist_laden,
                "from_busy": None,
                "to_busy": None,
                "sig_wave_height_m": None,
                "wind_speed_ms": None,
            })
            pred = ml_out.get("prediction", {})
            port_delay_days = float(
                pred.get("port_delay_days", pred.get("port_delay", 0.0)) or 0.0
            )
            weather_factor = float(ml_out.get("prediction", {}).get("weather_factor", 1.0) or 1.0)
            load_port_delay_days = float(pred.get("load_port_delay_days", 0.0) or 0.0)
            discharge_port_delay_days = float(pred.get("discharge_port_delay_days", 0.0) or 0.0)
            ml_risk = {
                "port_delay_days": port_delay_days,
                "load_port_delay_days": load_port_delay_days,
                "discharge_port_delay_days": discharge_port_delay_days,
                "weather_factor": weather_factor,
            }
        except Exception:
            ml_risk = None

    include_ballast = bool(scenario.include_ballast)
    cur_port = _norm_port(vessel.get("current_port", "")) if include_ballast else ""
    if include_ballast and cur_port:
        try:
            dist_ballast = get_distance_nm(port_distances, cur_port, load_port)
        except ValueError:
            dist_ballast = 0.0
            ballast_days = 0.0
        else:
            econ_speed_ballast = to_float_or_default(vessel.get("econ_speed_ballast_kn"), np.nan)
            if not np.isfinite(econ_speed_ballast) or econ_speed_ballast <= 0:
                raise ValueError("Invalid econ_speed_ballast_kn")
            ballast_days = (dist_ballast / (econ_speed_ballast * weather_factor)) / 24.0

    econ_speed_laden = to_float_or_default(vessel.get("econ_speed_laden_kn"), np.nan)
    if not np.isfinite(econ_speed_laden) or econ_speed_laden <= 0:
        raise ValueError("Invalid econ_speed_laden_kn")
    laden_days = (dist_laden / (econ_speed_laden * weather_factor)) / 24.0

    vlsf_laden_tpd = to_float_or_default(vessel.get("econ_fuel_laden_vlsf_mt_per_day"), np.nan)
    mgo_laden_tpd = to_float_or_default(vessel.get("econ_fuel_laden_mgo_mt_per_day"), np.nan)
    if not np.isfinite(vlsf_laden_tpd) or not np.isfinite(mgo_laden_tpd):
        raise ValueError("Invalid laden fuel consumption")

    vlsf_sea = vlsf_laden_tpd * laden_days
    mgo_sea = mgo_laden_tpd * laden_days

    if include_ballast and cur_port:
        vlsf_ballast_tpd = to_float_or_default(vessel.get("econ_fuel_ballast_vlsf_mt_per_day"), np.nan)
        mgo_ballast_tpd = to_float_or_default(vessel.get("econ_fuel_ballast_mgo_mt_per_day"), np.nan)
        if not np.isfinite(vlsf_ballast_tpd) or not np.isfinite(mgo_ballast_tpd):
            raise ValueError("Invalid ballast fuel consumption")
        vlsf_sea += vlsf_ballast_tpd * ballast_days
        mgo_sea += mgo_ballast_tpd * ballast_days

    qty = to_float_or_default(cargo.get("quantity_mt"), np.nan)
    tol = to_float_or_default(cargo.get("quantity_tolerance_pct"), 0.0)
    dwt = to_float_or_default(vessel.get("dwt_mt"), np.nan)
    if not np.isfinite(qty) or not np.isfinite(dwt):
        raise ValueError("Invalid qty/dwt")

    qty_min = qty * (1 - tol / 100.0)
    qty_max = min(qty * (1 + tol / 100.0), dwt)

    load_rate = to_float_or_default(cargo.get("load_rate_mt_per_day"), np.nan)
    dis_rate = to_float_or_default(cargo.get("discharge_rate_mt_per_day"), np.nan)
    if not np.isfinite(load_rate) or not np.isfinite(dis_rate) or load_rate <= 0 or dis_rate <= 0:
        raise ValueError("Invalid load/discharge rates")

    load_tt_days = to_float_or_default(cargo.get("load_turn_time_hr"), 0.0) / 24.0
    dis_tt_days = to_float_or_default(cargo.get("discharge_turn_time_hr"), 0.0) / 24.0

    load_days_min = qty_min / load_rate
    load_days_max = qty_max / load_rate
    dis_days_min = qty_min / dis_rate
    dis_days_max = qty_max / dis_rate

    load_wait = float(scenario.load_wait_days) + load_port_delay_days
    discharge_wait = float(scenario.discharge_wait_days) + discharge_port_delay_days
    if is_china_port(discharge_port):
        discharge_wait += float(scenario.china_port_delay_days)

    idle_days = load_tt_days + dis_tt_days + load_wait + discharge_wait

    total_days_min = (
        ballast_days + laden_days
        + load_days_min + dis_days_min
        + load_tt_days + dis_tt_days
        + load_wait + discharge_wait
    )
    total_days_max = (
        ballast_days + laden_days
        + load_days_max + dis_days_max
        + load_tt_days + dis_tt_days
        + load_wait + discharge_wait
    )

    port_working_vlsf = to_float_or_default(vessel.get("port_working_vlsf_mt_per_day"), 0.0)
    port_idle_mgo = to_float_or_default(vessel.get("port_idle_mgo_mt_per_day"), 0.0)

    working_min = load_days_min + dis_days_min
    working_max = load_days_max + dis_days_max
    port_vlsf_min = port_working_vlsf * working_min
    port_vlsf_max = port_working_vlsf * working_max
    port_mgo = port_idle_mgo * idle_days

    total_vlsf_min = vlsf_sea + port_vlsf_min
    total_vlsf_max = vlsf_sea + port_vlsf_max
    total_mgo = mgo_sea + port_mgo

    vlsf_price = get_bunker_price(
        bunker, scenario.bunker_location, "VLSFO", scenario.bunker_month_col,
        uplift_pct=scenario.bunker_price_uplift_pct
    )
    mgo_price = get_bunker_price(
        bunker, scenario.bunker_location, "MGO", scenario.bunker_month_col,
        uplift_pct=scenario.bunker_price_uplift_pct
    )

    bunker_cost_min = total_vlsf_min * vlsf_price + total_mgo * mgo_price
    bunker_cost_max = total_vlsf_max * vlsf_price + total_mgo * mgo_price

    hire_raw = vessel.get("hire_rate_usd_per_day")
    hire_rate = to_float_or_default(hire_raw, default=np.nan)
    if not np.isfinite(hire_rate) or hire_rate <= 0:
        if default_hire_usd_per_day is None:
            raise ValueError(f"Missing/invalid hire_rate_usd_per_day for {vessel.get('vessel_name')}: {hire_raw}")
        hire_rate = float(default_hire_usd_per_day)

    hire_cost_min = hire_rate * total_days_min
    hire_cost_max = hire_rate * total_days_max

    freight_rate = to_float_or_default(cargo.get("freight_rate_usd_per_mt"), np.nan)
    if not np.isfinite(freight_rate):
        raise ValueError("Invalid freight_rate_usd_per_mt")

    freight_revenue_min = qty_min * freight_rate
    freight_revenue_max = qty_max * freight_rate

    commission_pct = to_float_or_default(cargo.get("commission_pct"), 0.0) / 100.0
    commission_min = freight_revenue_min * commission_pct
    commission_max = freight_revenue_max * commission_pct

    load_port_cost = to_float_or_default(cargo.get("load_port_cost_usd"), 0.0)
    discharge_port_cost = to_float_or_default(cargo.get("discharge_port_cost_usd"), 0.0)
    port_costs = load_port_cost + discharge_port_cost

    profit_min = freight_revenue_min - commission_min - hire_cost_min - bunker_cost_min - port_costs
    profit_max = freight_revenue_max - commission_max - hire_cost_max - bunker_cost_max - port_costs

    tce_min = profit_min / total_days_min if total_days_min > 0 else np.nan
    tce_max = profit_max / total_days_max if total_days_max > 0 else np.nan

    return {
        "vessel_id": vessel.get("vessel_id"),
        "vessel_name": vessel.get("vessel_name"),
        "cargo_id": cargo.get("cargo_id"),
        "load_port": load_port,
        "discharge_port": discharge_port,
        "scenario": scenario.__dict__,
        "ml_risk": ml_risk,
        "dist_nm": {"ballast": dist_ballast, "laden": dist_laden},
        "qty": {"min_mt": qty_min, "max_mt": qty_max},
        "days": {
            "ballast_days": ballast_days,
            "laden_days": laden_days,
            "idle_days": idle_days,
            "total_days_min": total_days_min,
            "total_days_max": total_days_max,
        },
        "money": {
            "profit_mid_usd": (profit_min + profit_max) / 2.0,
            "tce_mid_usd_per_day": (tce_min + tce_max) / 2.0 if np.isfinite(tce_min) and np.isfinite(tce_max) else np.nan,
            "hire_rate_usd_per_day": hire_rate,
        },
    }


def evaluate_all(
    vessels: pd.DataFrame,
    cargoes: pd.DataFrame,
    port_distances: pd.DataFrame,
    bunker: pd.DataFrame,
    scenario: Scenario,
    *,
    default_hire_usd_per_day: Optional[float] = None,
) -> pd.DataFrame:
    vessels = normalize_df_columns(vessels)
    cargoes = normalize_df_columns(cargoes)
    port_distances = normalize_df_columns(port_distances)
    bunker = normalize_df_columns(bunker)

    rows = []
    for _, v in vessels.iterrows():
        for _, c in cargoes.iterrows():
            try:
                r = calc_voyage(v, c, port_distances, bunker, scenario, default_hire_usd_per_day=default_hire_usd_per_day)
                rows.append({
                    "vessel_id": r["vessel_id"],
                    "vessel_name": r["vessel_name"],
                    "cargo_id": r["cargo_id"],
                    "load_port": r["load_port"],
                    "discharge_port": r["discharge_port"],
                    "profit_mid_usd": r["money"]["profit_mid_usd"],
                    "tce_mid_usd_per_day": r["money"]["tce_mid_usd_per_day"],
                    "error": None,
                })
            except Exception as e:
                rows.append({
                    "vessel_id": v.get("vessel_id"),
                    "vessel_name": v.get("vessel_name"),
                    "cargo_id": c.get("cargo_id"),
                    "load_port": c.get("load_port"),
                    "discharge_port": c.get("discharge_port"),
                    "profit_mid_usd": np.nan,
                    "tce_mid_usd_per_day": np.nan,
                    "error": str(e),
                })
    return pd.DataFrame(rows)


def main():
    from portfolio_optimizer import optimize_portfolio
    from ml import RiskModel

    cargill_vessels, cargill_committed, market_vessels, market_cargoes, port_distances, ffa, bunker = data_in.load_data()

    risk_model = RiskModel()

    scenario = Scenario(
        bunker_location="Durban",
        bunker_month_col="Apr_26",
        load_wait_days=0.0,
        discharge_wait_days=0.0,
        include_ballast=True,
        china_port_delay_days=0.0,
        bunker_price_uplift_pct=5.0,
    )

    try:
        plan_df, summary = optimize_portfolio(
            cargill_vessels=cargill_vessels,
            cargill_committed=cargill_committed,
            market_vessels=market_vessels,
            market_cargoes=market_cargoes,
            port_distances=port_distances,
            bunker=bunker,
            scenario=scenario,
            risk_model=risk_model,
            default_market_hire_usd_per_day=25000.0,
            metric="profit_mid_usd",
        )
        def _fmt(df: pd.DataFrame) -> str:
            fmt = {
                "profit_mid_usd": "{:,.0f}".format,
                "tce_mid_usd_per_day": "{:,.0f}".format,
                "hire_rate_usd_per_day": "{:,.0f}".format,
                "ml_port_delay_days": "{:.2f}".format,
                "ml_load_port_delay_days": "{:.2f}".format,
                "ml_discharge_port_delay_days": "{:.2f}".format,
                "ml_weather_factor": "{:.3f}".format,
            }
            return df.to_string(index=False, formatters={k: v for k, v in fmt.items() if k in df.columns})

        print("\n=== OPTIMAL PORTFOLIO PLAN ===")
        print(_fmt(plan_df))

        print("\n=== SUMMARY ===")
        for k, v in summary.items():
            print(f"{k}: {v}")

    except RuntimeError as e:
        print("\n[ERROR]", e)
        print("\n--- Debug: committed feasibility check (top errors) ---")

        for flag in [True, False]:
            sc = Scenario(**scenario.__dict__)
            sc.include_ballast = flag
            df = evaluate_all(
                vessels=cargill_vessels,
                cargoes=cargill_committed,
                port_distances=port_distances,
                bunker=bunker,
                scenario=sc,
                default_hire_usd_per_day=None,
            )
            feasible = df[df["error"].isna()]
            print(f"\ninclude_ballast={flag}: feasible={len(feasible)} / total={len(df)}")

            err_counts = df["error"].dropna().value_counts().head(10)
            if len(err_counts):
                print("\nTop errors:")
                for msg, cnt in err_counts.items():
                    print(f"- {cnt}x {msg}")
            else:
                print("\nNo errors captured (unexpected).")

        raise


if __name__ == "__main__":
    main()
