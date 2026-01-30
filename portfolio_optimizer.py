import itertools
import numpy as np
import pandas as pd

from cal_gpt import calc_voyage
from ml import RiskModel   # ✅ NEW


def _key(vessel_id: str, cargo_id: str) -> tuple:
    return (str(vessel_id), str(cargo_id))


def build_result_map(
    vessels: pd.DataFrame,
    cargoes: pd.DataFrame,
    port_distances: pd.DataFrame,
    bunker: pd.DataFrame,
    scenario,
    *,
    risk_model=None,  # ✅ NEW
    default_hire_usd_per_day=None,
    metric="profit_mid_usd",
):
    """
    Return dict: (vessel_id, cargo_id) -> full result dict from calc_voyage()
    """
    out = {}
    for _, v in vessels.iterrows():
        for _, c in cargoes.iterrows():
            try:
                r = calc_voyage(
                    v, c, port_distances, bunker, scenario,
                    risk_model=risk_model,  # ✅ NEW (calc_voyage must accept this)
                    default_hire_usd_per_day=default_hire_usd_per_day
                )
                out[_key(r["vessel_id"], r["cargo_id"])] = r
            except Exception:
                continue
    return out


def best_market_fill_for_unused_cargill(
    unused_cargill_vessel_ids,
    market_cargo_ids,
    cargill_market_map,
    metric="profit_mid_usd",
):
    unused_v = list(unused_cargill_vessel_ids)
    mkt_c = list(market_cargo_ids)

    best_extra = []
    best_value = 0.0

    max_k = min(len(unused_v), len(mkt_c))
    for k in range(max_k + 1):
        for cargos_subset in itertools.combinations(mkt_c, k):
            for perm in itertools.permutations(cargos_subset, k):
                total = 0.0
                picks = []
                feasible = True
                for vid, cid in zip(unused_v, perm):
                    r = cargill_market_map.get(_key(vid, cid))
                    if r is None:
                        feasible = False
                        break
                    total += float(r["money"][metric])
                    picks.append(r)
                if feasible and total > best_value:
                    best_value = total
                    best_extra = picks

    return best_extra, best_value


def optimize_portfolio(
    cargill_vessels: pd.DataFrame,
    cargill_committed: pd.DataFrame,
    market_vessels: pd.DataFrame,
    market_cargoes: pd.DataFrame,
    port_distances: pd.DataFrame,
    bunker: pd.DataFrame,
    scenario,
    *,
    risk_model=None,  # ✅ NEW
    default_market_hire_usd_per_day=25000.0,
    metric="profit_mid_usd",
):
    """
    최종 목표:
    - committed cargo 3개는 무조건 커버
    - 커버는 (Cargill vessel or Market vessel) 가능
    - 사용한 vessel은 중복 사용 금지
    - 남는 Cargill vessel은 market cargo로 추가 수익 배정
    """

    # --- 1) 필요한 조합 3종의 결과맵 만들기 ---
    cargill_committed_map = build_result_map(
        cargill_vessels, cargill_committed, port_distances, bunker, scenario,
        risk_model=risk_model,  # ✅ NEW
        default_hire_usd_per_day=None,
        metric=metric
    )

    cargill_market_map = build_result_map(
        cargill_vessels, market_cargoes, port_distances, bunker, scenario,
        risk_model=risk_model,  # ✅ NEW
        default_hire_usd_per_day=None,
        metric=metric
    )

    market_committed_map = build_result_map(
        market_vessels, cargill_committed, port_distances, bunker, scenario,
        risk_model=risk_model,  # ✅ NEW
        default_hire_usd_per_day=default_market_hire_usd_per_day,
        metric=metric
    )

    committed_ids = list(cargill_committed["cargo_id"].astype(str))
    cargill_vessel_ids = set(cargill_vessels["vessel_id"].astype(str))
    market_vessel_ids = set(market_vessels["vessel_id"].astype(str))
    market_cargo_ids = set(market_cargoes["cargo_id"].astype(str))

    candidates_by_cargo = {}
    for cid in committed_ids:
        candidates = []

        for vid in cargill_vessel_ids:
            r = cargill_committed_map.get(_key(vid, cid))
            if r is not None and np.isfinite(r["money"][metric]):
                candidates.append(("cargill", vid, r))

        for vid in market_vessel_ids:
            r = market_committed_map.get(_key(vid, cid))
            if r is not None and np.isfinite(r["money"][metric]):
                candidates.append(("market", vid, r))

        candidates_by_cargo[cid] = candidates

    best_plan_committed = None
    best_total = -1e18

    cargo0, cargo1, cargo2 = committed_ids[0], committed_ids[1], committed_ids[2]

    for c0 in candidates_by_cargo[cargo0]:
        for c1 in candidates_by_cargo[cargo1]:
            for c2 in candidates_by_cargo[cargo2]:
                used_vessels = {c0[1], c1[1], c2[1]}
                if len(used_vessels) < 3:
                    continue

                committed_picks = [c0[2], c1[2], c2[2]]
                committed_profit = sum(float(r["money"][metric]) for r in committed_picks)

                used_cargill = {x[1] for x in [c0, c1, c2] if x[0] == "cargill"}
                unused_cargill = sorted(list(cargill_vessel_ids - used_cargill))

                extra_picks, extra_profit = best_market_fill_for_unused_cargill(
                    unused_cargill, market_cargo_ids, cargill_market_map, metric=metric
                )

                total = committed_profit + extra_profit
                if total > best_total:
                    best_total = total
                    best_plan_committed = committed_picks
                    best_plan_extra = extra_picks

    if best_plan_committed is None:
        raise RuntimeError("No feasible committed assignment found (check data/ports/hire).")

    plan_rows = []
    for r in best_plan_committed:
        plan_rows.append({
            "type": "COMMITTED",
            "vessel_id": r["vessel_id"],
            "vessel_name": r["vessel_name"],
            "cargo_id": r["cargo_id"],
            "load_port": r["load_port"],
            "discharge_port": r["discharge_port"],
            "profit_mid_usd": r["money"]["profit_mid_usd"],
            "tce_mid_usd_per_day": r["money"]["tce_mid_usd_per_day"],
            "hire_rate_usd_per_day": r["money"]["hire_rate_usd_per_day"],
            # ✅ Optional: show ML adjustments
            "ml_port_delay_days": (r.get("ml_risk") or {}).get("port_delay_days", np.nan),
            "ml_load_port_delay_days": (r.get("ml_risk") or {}).get("load_port_delay_days", np.nan),
            "ml_discharge_port_delay_days": (r.get("ml_risk") or {}).get("discharge_port_delay_days", np.nan),
            "ml_weather_factor": (r.get("ml_risk") or {}).get("weather_factor", np.nan),
        })

    for r in best_plan_extra:
        plan_rows.append({
            "type": "MARKET_FILL",
            "vessel_id": r["vessel_id"],
            "vessel_name": r["vessel_name"],
            "cargo_id": r["cargo_id"],
            "load_port": r["load_port"],
            "discharge_port": r["discharge_port"],
            "profit_mid_usd": r["money"]["profit_mid_usd"],
            "tce_mid_usd_per_day": r["money"]["tce_mid_usd_per_day"],
            "hire_rate_usd_per_day": r["money"]["hire_rate_usd_per_day"],
            "ml_port_delay_days": (r.get("ml_risk") or {}).get("port_delay_days", np.nan),
            "ml_load_port_delay_days": (r.get("ml_risk") or {}).get("load_port_delay_days", np.nan),
            "ml_discharge_port_delay_days": (r.get("ml_risk") or {}).get("discharge_port_delay_days", np.nan),
            "ml_weather_factor": (r.get("ml_risk") or {}).get("weather_factor", np.nan),
        })

    plan_df = pd.DataFrame(plan_rows).sort_values(
        ["type", "profit_mid_usd"],
        ascending=[True, False]
    ).reset_index(drop=True)

    summary = {
        "objective_metric": metric,
        "total_profit_mid_usd": float(plan_df["profit_mid_usd"].sum()),
        "committed_profit_mid_usd": float(plan_df[plan_df["type"] == "COMMITTED"]["profit_mid_usd"].sum()),
        "market_fill_profit_mid_usd": float(plan_df[plan_df["type"] == "MARKET_FILL"]["profit_mid_usd"].sum()),
        "num_committed": int((plan_df["type"] == "COMMITTED").sum()),
        "num_market_fill": int((plan_df["type"] == "MARKET_FILL").sum()),
    }

    return plan_df, summary
