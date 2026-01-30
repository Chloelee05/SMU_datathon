from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False


def _rf(cat_cols: list[str], num_cols: list[str], *, n: int, seed: int, min_leaf: int = 2) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=n,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=min_leaf,
    )
    return Pipeline([("pre", pre), ("model", model)])


def _names(pipe: Pipeline) -> list[str]:
    return pipe.named_steps["pre"].get_feature_names_out().tolist()


def top_importance(pipe: Pipeline, top_k: int = 5) -> list[dict]:
    names = _names(pipe)
    imp = pipe.named_steps["model"].feature_importances_
    idx = np.argsort(imp)[::-1][:top_k]
    return [{"feature": names[i], "importance": round(float(imp[i]), 3)} for i in idx]


def top_shap(pipe: Pipeline, X_one: pd.DataFrame, bg_X: pd.DataFrame, top_k: int = 5) -> list[dict] | None:
    if not HAS_SHAP or bg_X is None or len(bg_X) == 0:
        return None
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    bg = pre.transform(bg_X)
    x = pre.transform(X_one)
    explainer = shap.TreeExplainer(model, bg)
    sv = explainer.shap_values(x)
    if isinstance(sv, list):
        sv = sv[0]
    vals = np.asarray(sv).reshape(-1)
    names = _names(pipe)
    idx = np.argsort(np.abs(vals))[::-1][:top_k]
    return [{"feature": names[i], "impact": round(float(vals[i]), 3)} for i in idx]


def _clean_ports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["from_port", "to_port"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(1).astype(int).clip(1, 12)
    return df


def _safe_num(s: pd.Series, default: float) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _stable_seed(*parts: object) -> int:
    s = "|".join("" if p is None else str(p) for p in parts)
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False) & 0x7FFFFFFF


class RiskModel:
    def __init__(self, *, port_csv: str = "port_delay.csv", weather_csv: str = "weather_delay.csv", bg_n: int = 300):
        self.port_csv = port_csv
        self.weather_csv = weather_csv
        self.bg_n = int(bg_n)
        self._trained = False

        self.cat = ["from_port", "to_port"]
        self.port_num = ["month", "distance_nm", "from_busy", "to_busy"]
        self.weather_num = ["month", "distance_nm", "sig_wave_height_m", "wind_speed_ms"]

        self.port_model: Pipeline | None = None
        self.weather_model: Pipeline | None = None
        self.port_bg_X: pd.DataFrame | None = None
        self.weather_bg_X: pd.DataFrame | None = None

        self.port_raw: pd.DataFrame | None = None
        self.weather_raw: pd.DataFrame | None = None

        self.port_medians: dict[str, float] = {}
        self.weather_medians: dict[str, float] = {}

        self.busy_from_by_port_month: dict[tuple[str, int], float] = {}
        self.busy_to_by_port_month: dict[tuple[str, int], float] = {}
        self.weather_by_month: dict[int, pd.DataFrame] = {}

        self.port_delay_cap = 15.0
        self.weather_extra_cap = 6.0

    def _train_once(self) -> None:
        if self._trained:
            return

        port = _clean_ports(pd.read_csv(self.port_csv))
        weather = _clean_ports(pd.read_csv(self.weather_csv))

        port["distance_nm"] = _safe_num(port.get("distance_nm", pd.Series([], dtype=float)), 3000.0).clip(0, 25000)
        port["from_busy"] = _safe_num(port.get("from_busy", pd.Series([], dtype=float)), 0.5).clip(0, 1)
        port["to_busy"] = _safe_num(port.get("to_busy", pd.Series([], dtype=float)), 0.5).clip(0, 1)
        port["port_delay_days"] = _safe_num(port.get("port_delay_days", pd.Series([], dtype=float)), 0.0).clip(0, self.port_delay_cap)

        weather["distance_nm"] = _safe_num(weather.get("distance_nm", pd.Series([], dtype=float)), 3000.0).clip(0, 25000)
        weather["sig_wave_height_m"] = _safe_num(weather.get("sig_wave_height_m", pd.Series([], dtype=float)), 2.0).clip(0, 20)
        weather["wind_speed_ms"] = _safe_num(weather.get("wind_speed_ms", pd.Series([], dtype=float)), 8.0).clip(0, 60)
        weather["weather_delay_days"] = _safe_num(weather.get("weather_delay_days", pd.Series([], dtype=float)), 0.0).clip(0, self.weather_extra_cap)

        self.port_raw = port
        self.weather_raw = weather

        self.port_medians = {
            "month": float(port["month"].median()) if "month" in port.columns and len(port) else 7.0,
            "distance_nm": float(port["distance_nm"].median()) if "distance_nm" in port.columns and len(port) else 3000.0,
            "from_busy": float(port["from_busy"].median()) if "from_busy" in port.columns and len(port) else 0.5,
            "to_busy": float(port["to_busy"].median()) if "to_busy" in port.columns and len(port) else 0.5,
        }
        self.weather_medians = {
            "month": float(weather["month"].median()) if "month" in weather.columns and len(weather) else 7.0,
            "distance_nm": float(weather["distance_nm"].median()) if "distance_nm" in weather.columns and len(weather) else 3000.0,
            "sig_wave_height_m": float(weather["sig_wave_height_m"].median()) if "sig_wave_height_m" in weather.columns and len(weather) else 2.0,
            "wind_speed_ms": float(weather["wind_speed_ms"].median()) if "wind_speed_ms" in weather.columns and len(weather) else 8.0,
        }

        if len(port):
            g1 = port.groupby(["from_port", "month"], dropna=False)["from_busy"].median()
            self.busy_from_by_port_month = {(str(p), int(m)): float(v) for (p, m), v in g1.items()}

            g2 = port.groupby(["to_port", "month"], dropna=False)["to_busy"].median()
            self.busy_to_by_port_month = {(str(p), int(m)): float(v) for (p, m), v in g2.items()}

        if len(weather):
            for m in range(1, 13):
                w_m = weather[weather["month"] == m]
                self.weather_by_month[m] = w_m if len(w_m) else weather

        port_X = port[self.cat + self.port_num].copy()
        port_y = port["port_delay_days"].astype(float)

        weather_X = weather[self.cat + self.weather_num].copy()
        weather_y = weather["weather_delay_days"].astype(float)

        self.port_model = _rf(self.cat, self.port_num, n=700, seed=11, min_leaf=2).fit(port_X, port_y)
        self.weather_model = _rf(self.cat, self.weather_num, n=700, seed=22, min_leaf=2).fit(weather_X, weather_y)

        self.port_bg_X = port_X.sample(n=min(self.bg_n, len(port_X)), random_state=7) if len(port_X) else port_X
        self.weather_bg_X = weather_X.sample(n=min(self.bg_n, len(weather_X)), random_state=8) if len(weather_X) else weather_X

        self._trained = True

    def _pick_month(self, month) -> int:
        m = pd.to_numeric(month, errors="coerce")
        if m is None or not np.isfinite(m):
            m = self.port_medians["month"]
        m = int(m)
        return int(np.clip(m, 1, 12))

    def _pick_distance(self, distance_nm) -> float:
        d = pd.to_numeric(distance_nm, errors="coerce")
        if d is None or not np.isfinite(d):
            d = self.port_medians["distance_nm"]
        return float(np.clip(float(d), 0.0, 25000.0))

    def _simulate_congestion(self, from_port: str, to_port: str, month: int, distance_nm: float, seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        fb = self.busy_from_by_port_month.get((from_port, month), self.port_medians["from_busy"])
        tb = self.busy_to_by_port_month.get((to_port, month), self.port_medians["to_busy"])
        route_scale = np.clip(distance_nm / 7000.0, 0.0, 1.5)
        peak = month in (11, 12, 1, 2)
        season = 0.06 if peak else 0.0
        sigma = 0.06 + 0.03 * route_scale
        fb = float(np.clip(fb + season + rng.normal(0, sigma), 0.0, 1.0))
        tb = float(np.clip(tb + season + rng.normal(0, sigma), 0.0, 1.0))
        return fb, tb

    def _simulate_weather(self, month: int, distance_nm: float, seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        w = self.weather_by_month.get(month, self.weather_raw if self.weather_raw is not None else pd.DataFrame())
        if w is not None and len(w) >= 10:
            row = w.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            hs = float(row.get("sig_wave_height_m", self.weather_medians["sig_wave_height_m"]))
            ws = float(row.get("wind_speed_ms", self.weather_medians["wind_speed_ms"]))
        else:
            season = float(np.cos((month - 1) / 12.0 * 2.0 * np.pi))
            hs = float(1.4 + 0.6 * season + 0.00012 * distance_nm)
            ws = float(4.5 + 2.1 * hs)

        route_scale = np.clip(distance_nm / 6000.0, 0.0, 1.5)
        hs = float(max(0.0, hs + rng.normal(0, 0.25 + 0.12 * route_scale)))
        ws = float(max(0.0, ws + rng.normal(0, 1.0 + 0.45 * route_scale)))
        return hs, ws

    def predict(self, inputs: dict) -> dict:
        self._train_once()
        assert self.port_model is not None and self.weather_model is not None
        assert self.port_bg_X is not None and self.weather_bg_X is not None

        from_port = str(inputs.get("from_port", "")).upper().strip()
        to_port = str(inputs.get("to_port", "")).upper().strip()
        if not from_port or not to_port:
            raise ValueError("from_port and to_port are required")

        month = self._pick_month(inputs.get("month", None))
        distance_nm = self._pick_distance(inputs.get("distance_nm", None))

        fb = inputs.get("from_busy", None)
        tb = inputs.get("to_busy", None)
        hs = inputs.get("sig_wave_height_m", None)
        ws = inputs.get("wind_speed_ms", None)

        sim_seed = _stable_seed(from_port, to_port, month, int(round(distance_nm)))

        if fb is None or tb is None:
            fb_s, tb_s = self._simulate_congestion(from_port, to_port, month, distance_nm, seed=sim_seed ^ 0xA5A5)
            fb = fb_s if fb is None else float(np.clip(float(fb), 0.0, 1.0))
            tb = tb_s if tb is None else float(np.clip(float(tb), 0.0, 1.0))
        else:
            fb = float(np.clip(float(fb), 0.0, 1.0))
            tb = float(np.clip(float(tb), 0.0, 1.0))

        if hs is None or ws is None:
            hs_s, ws_s = self._simulate_weather(month, distance_nm, seed=sim_seed ^ 0x5A5A)
            hs = hs_s if hs is None else float(max(0.0, float(hs)))
            ws = ws_s if ws is None else float(max(0.0, float(ws)))
        else:
            hs = float(max(0.0, float(hs)))
            ws = float(max(0.0, float(ws)))

        Xp = pd.DataFrame(
            [
                {
                    "from_port": from_port,
                    "to_port": to_port,
                    "month": month,
                    "distance_nm": distance_nm,
                    "from_busy": fb,
                    "to_busy": tb,
                }
            ]
        )

        Xw = pd.DataFrame(
            [
                {
                    "from_port": from_port,
                    "to_port": to_port,
                    "month": month,
                    "distance_nm": distance_nm,
                    "sig_wave_height_m": hs,
                    "wind_speed_ms": ws,
                }
            ]
        )

        port_delay_days = float(np.clip(self.port_model.predict(Xp)[0], 0.0, self.port_delay_cap))
        weather_extra_sea_days = float(np.clip(self.weather_model.predict(Xw)[0], 0.0, self.weather_extra_cap))

        scale = (distance_nm / 3000.0) + 0.5
        if weather_extra_sea_days >= 4.5:
            lo, hi = 0.85, 0.95
            cap = self.weather_extra_cap
        elif weather_extra_sea_days >= 2.5:
            lo, hi = 0.90, 0.98
            cap = 4.5
        else:
            lo, hi = 0.95, 1.00
            cap = 2.5
        t = 1.0 - (weather_extra_sea_days / max(1e-6, cap))
        t = float(np.clip(t, 0.0, 1.0))
        base = lo + (hi - lo) * t
        rng = np.random.default_rng(sim_seed ^ 0xC3C3)
        jitter = float(rng.normal(0.0, 0.004))
        weather_factor = float(np.clip(base + jitter, lo, hi))

        port_delay_days = float(np.round(port_delay_days, 2))
        if port_delay_days > 0:
            denom = max(1e-6, float(fb) + float(tb))
            load_port_delay_days = float(np.round(port_delay_days * (float(fb) / denom), 2))
            discharge_port_delay_days = float(np.round(port_delay_days * (float(tb) / denom), 2))
        else:
            load_port_delay_days = 0.0
            discharge_port_delay_days = 0.0
        weather_extra_sea_days = float(np.round(weather_extra_sea_days, 2))
        weather_factor = float(np.round(weather_factor, 3))

        fi_port = top_importance(self.port_model, top_k=5)
        fi_weather = top_importance(self.weather_model, top_k=5)

        shap_port = top_shap(self.port_model, Xp, self.port_bg_X, top_k=5)
        shap_weather = top_shap(self.weather_model, Xw, self.weather_bg_X, top_k=5)

        return {
            "prediction": {
                "port_delay_days": port_delay_days,
                "load_port_delay_days": load_port_delay_days,
                "discharge_port_delay_days": discharge_port_delay_days,
                "weather_factor": weather_factor,
            },
            "explanation": {
                "inputs_used": {
                    "from_port": from_port,
                    "to_port": to_port,
                    "month": month,
                    "distance_nm": round(distance_nm, 2),
                    "from_busy": round(float(fb), 3),
                    "to_busy": round(float(tb), 3),
                    "sig_wave_height_m": round(float(hs), 3),
                    "wind_speed_ms": round(float(ws), 3),
                    "sim_seed": int(sim_seed),
                },
                "feature_importance": {
                    "port_delay_days": fi_port,
                    "weather_extra_sea_days": fi_weather,
                },
                "shap_values": {
                    "port_delay_days": shap_port,
                    "weather_extra_sea_days": shap_weather,
                },
            },
        }
