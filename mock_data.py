import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

ports = [
    "SINGAPORE", "QINGDAO", "ROTTERDAM", "SANTOS",
    "RICHARDS BAY", "NEWCASTLE", "DALRYMPLE BAY"
]

rows = []

for _ in range(3000):
    from_port = rng.choice(ports)
    to_port = rng.choice([p for p in ports if p != from_port])
    month = rng.integers(1, 13)
    distance_nm = rng.uniform(1500, 12000)

    from_busy = rng.uniform(0, 1)
    to_busy = rng.uniform(0, 1)

    # ---- PORT DELAY (days) ----
    port_delay_days = (
        0.4
        + 2.0 * from_busy
        + 2.0 * to_busy
        + 0.00005 * distance_nm
        + rng.normal(0, 0.4)
    )
    port_delay_days = max(0, min(port_delay_days, 10))

    # ---- WEATHER ----
    sig_wave_height_m = (
        1.5
        + 1.5 * (month in [12, 1, 2])
        + 0.0001 * distance_nm
        + rng.normal(0, 0.3)
    )
    sig_wave_height_m = max(0.5, sig_wave_height_m)

    wind_speed_ms = 4 + 2 * sig_wave_height_m + rng.normal(0, 1)

    weather_delay_days = (
        0.06 * sig_wave_height_m**2
        + 0.02 * wind_speed_ms
        + rng.normal(0, 0.3)
    )
    weather_delay_days = max(0, min(weather_delay_days, 8))

    rows.append([
        from_port, to_port, month, distance_nm,
        from_busy, to_busy, port_delay_days,
        sig_wave_height_m, wind_speed_ms, weather_delay_days
    ])

df = pd.DataFrame(rows, columns=[
    "from_port", "to_port", "month", "distance_nm",
    "from_busy", "to_busy", "port_delay_days",
    "sig_wave_height_m", "wind_speed_ms", "weather_delay_days"
])

df[[
    "from_port", "to_port", "month", "distance_nm",
    "from_busy", "to_busy", "port_delay_days"
]].to_csv("port_delay.csv", index=False)

df[[
    "from_port", "to_port", "month", "distance_nm",
    "sig_wave_height_m", "wind_speed_ms", "weather_delay_days"
]].to_csv("weather_delay.csv", index=False)

print("Generated port_delay.csv and weather_delay.csv")