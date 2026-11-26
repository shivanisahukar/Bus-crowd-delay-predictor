"""Train the delay and crowd models for the Bus Crowd & Delay Predictor.

NOTE: This prototype trains on SYNTHETIC data generated below, because no real
trip data is available yet. Replace make_data() with a loader for real GPS,
ticketing and weather records (for example GTFS-Realtime logs) to get real
predictions.

Run:  python train.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

ROUTE_BASE_DELAY = {1: 2.0, 2: 4.0, 3: 1.0}      # 1 = Campus, 2 = Market Loop, 3 = Highway Express
ROUTE_CROWD_BIAS = {1: 0.05, 2: 0.12, 3: 0.0}
FEATURES = ["route_id", "hour", "weekday", "rain"]
CROWD_LABELS = ["low", "medium", "high"]


def peak_factor(hour):
    """1.0 at the 8:00 and 17:30 peaks, near 0 at night."""
    morning = np.exp(-((hour - 8.0) ** 2) / 4.0)
    evening = np.exp(-((hour - 17.5) ** 2) / 5.0)
    return np.clip(morning + evening, 0, 1)


def make_data(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "route_id": rng.integers(1, 4, n),
        "hour": rng.integers(5, 23, n),
        "weekday": rng.integers(0, 7, n),      # 0 = Monday
        "rain": rng.integers(0, 2, n),
    })
    peak = peak_factor(df["hour"].to_numpy())
    weekend = (df["weekday"] >= 5).to_numpy()
    base = df["route_id"].map(ROUTE_BASE_DELAY).to_numpy()
    delay = base + 6 * peak + 4 * df["rain"] - 2 * weekend + rng.normal(0, 1.2, n)
    df["delay_min"] = np.clip(delay, 0, None).round(1)

    crowd = (0.15 + 0.65 * peak + 0.08 * df["rain"]
             + df["route_id"].map(ROUTE_CROWD_BIAS).to_numpy()
             - 0.15 * weekend + rng.normal(0, 0.07, n))
    df["crowd"] = pd.cut(crowd, [-np.inf, 0.4, 0.7, np.inf], labels=CROWD_LABELS).astype(str)
    return df


def main():
    df = make_data()
    X, y_delay, y_crowd = df[FEATURES], df["delay_min"], df["crowd"]
    Xtr, Xte, dtr, dte, ctr, cte = train_test_split(X, y_delay, y_crowd, test_size=0.2, random_state=0)

    delay_model = GradientBoostingRegressor(random_state=0).fit(Xtr, dtr)
    crowd_model = RandomForestClassifier(n_estimators=150, random_state=0).fit(Xtr, ctr)

    print(f"Delay model MAE:      {mean_absolute_error(dte, delay_model.predict(Xte)):.2f} min")
    print(f"Crowd model accuracy: {accuracy_score(cte, crowd_model.predict(Xte)):.2%}")
    print("(Scores are on synthetic data and say nothing about real-world accuracy.)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(delay_model, "models/delay_model.joblib")
    joblib.dump(crowd_model, "models/crowd_model.joblib")
    print("Saved models to the models/ folder.")


if __name__ == "__main__":
    main()
