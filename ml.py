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


def rf_pipeline(cat_cols, num_cols, *, n=300, seed=1):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    return Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=n, random_state=seed))])


def _names(pipe: Pipeline) -> list[str]:
    return pipe.named_steps["pre"].get_feature_names_out().tolist()


def top_importance(pipe: Pipeline, top_k: int = 3) -> list[dict]:
    names = _names(pipe)
    imp = pipe.named_steps["model"].feature_importances_
    idx = np.argsort(imp)[::-1][:top_k]
    return [
        {
            "feature": names[i],
            "importance": round(float(imp[i]), 3)
        }
        for i in idx
    ]


def top_shap(pipe: Pipeline, X_one: pd.DataFrame, bg: pd.DataFrame, top_k: int = 3) -> list[dict] | None:
    if not HAS_SHAP or bg is None or len(bg) == 0:
        return None

    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model, pre.transform(bg))
    sv = explainer.shap_values(pre.transform(X_one))

    if isinstance(sv, list):
        sv = sv[0]

    vals = np.asarray(sv).reshape(-1)
    names = _names(pipe)
    idx = np.argsort(np.abs(vals))[::-1][:top_k]

    return [
        {
            "feature": names[i],
            "impact": round(float(vals[i]), 3)
        }
        for i in idx
    ]


class RiskModel:
    def __init__(self, *, bg_n: int = 200,
                 port_csv: str = "port_delay.csv",
                 weather_csv: str = "weather_delay.csv"):
        self.bg_n = bg_n
        self.port_csv = port_csv
        self.weather_csv = weather_csv

        self._trained = False

        self.cat = ["from_port", "to_port"]
        self.port_num = ["month", "distance_nm", "from_busy", "to_busy"]
        self.weather_num = ["month", "distance_nm", "sig_wave_height_m", "wind_speed_ms"]

        self.port_rf: Pipeline | None = None
        self.weather_rf: Pipeline | None = None
        self.port_bg: pd.DataFrame | None = None
        self.weather_bg: pd.DataFrame | None = None

    def _train_once(self):
        if self._trained:
            return

        port = pd.read_csv(self.port_csv)
        weather = pd.read_csv(self.weather_csv)

        port_X = port[self.cat + self.port_num].copy()
        port_y = port["port_delay_days"].astype(float)

        self.port_rf = rf_pipeline(self.cat, self.port_num, seed=1).fit(port_X, port_y)
        self.port_bg = port_X.sample(n=min(self.bg_n, len(port_X)), random_state=7) if len(port_X) else port_X

        weather_X = weather[self.cat + self.weather_num].copy()

        d = pd.to_numeric(weather["weather_delay_days"], errors="coerce").fillna(0.0)
        weather_y = (1.0 - 0.02 * d).clip(0.88, 1.0)

        self.weather_rf = rf_pipeline(self.cat, self.weather_num, seed=2).fit(weather_X, weather_y)
        self.weather_bg = weather_X.sample(n=min(self.bg_n, len(weather_X)), random_state=8) if len(weather_X) else weather_X

        self._trained = True

    def predict(self, inputs: dict):
        self._train_once()

        assert self.port_rf is not None and self.weather_rf is not None
        assert self.port_bg is not None and self.weather_bg is not None

        port_inputs = {k: inputs[k] for k in (self.cat + self.port_num)}
        weather_inputs = {k: inputs.get(k) for k in (self.cat + self.weather_num)}

        if weather_inputs["sig_wave_height_m"] is None:
            weather_inputs["sig_wave_height_m"] = float(self.weather_bg["sig_wave_height_m"].mean())
        if weather_inputs["wind_speed_ms"] is None:
            weather_inputs["wind_speed_ms"] = float(self.weather_bg["wind_speed_ms"].mean())

        Xp = pd.DataFrame([port_inputs])
        Xw = pd.DataFrame([weather_inputs])

        port_delay = float(np.round(np.clip(self.port_rf.predict(Xp)[0], 0.0, 10.0), 2))
        weather_factor = float(np.round(np.clip(self.weather_rf.predict(Xw)[0], 0.88, 1.0), 3))

        fi_port = top_importance(self.port_rf)
        fi_weather = top_importance(self.weather_rf)

        shap_port = top_shap(self.port_rf, Xp, self.port_bg)
        shap_weather = top_shap(self.weather_rf, Xw, self.weather_bg)

        return {
            "prediction": {
                "port_delay": port_delay,
                "weather_factor": weather_factor,
            },
            "explanation": {
                "feature_importance": {
                    "port": fi_port,
                    "weather": fi_weather,
                },
                "shap_values": {
                    "port": shap_port,
                    "weather": shap_weather,
                },
            },
        }

