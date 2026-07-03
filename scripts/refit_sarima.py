import copy
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

SARIMA_ORDER = (1, 0, 1)
SARIMA_SEASONAL_ORDER = (1, 0, 1, 12)

FROZEN_MODELS = [
    {"name": "RF Sin TDA", "mae": 41.1941, "rmse": 52.1049, "r2": -0.1312, "improvement_vs_baseline": None, "live": False},
    {"name": "RF + TDA", "mae": 41.0917, "rmse": 52.0403, "r2": -0.1284, "improvement_vs_baseline": 0.25, "live": False},
    {"name": "SARIMA", "mae": 29.4225, "rmse": 36.1094, "r2": 0.4567, "improvement_vs_baseline": 28.58, "live": True},
    {"name": "SARIMA+TDA", "mae": 30.5655, "rmse": 36.4237, "r2": 0.4472, "improvement_vs_baseline": 25.80, "live": False},
    {"name": "LSTM+TDA", "mae": 29.4890, "rmse": 37.5017, "r2": 0.5687, "improvement_vs_baseline": 28.41, "live": False},
    {"name": "LSTM+TDA v2+Exog+Attention", "mae": 29.1069, "rmse": 38.6295, "r2": 0.5424, "improvement_vs_baseline": 29.34, "live": False},
]


def _fit_sarima(values):
    model = SARIMAX(
        values,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def compute_forecast(csv_path: str, backtest_months: int = 16, horizon_months: int = 12) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    min_required = backtest_months + 24  # need enough history to fit a seasonal model before the backtest split
    if len(df) < min_required:
        raise ValueError(f"series must have at least {min_required} months, got {len(df)}")

    values = df["WPUSI01102B"].values.astype(float)
    dates = df["observation_date"]

    # Backtest: hold out the last `backtest_months` to score the live metrics
    train_values = values[:-backtest_months]
    test_values = values[-backtest_months:]
    backtest_fit = _fit_sarima(train_values)
    backtest_preds = backtest_fit.forecast(steps=backtest_months)

    mae = float(mean_absolute_error(test_values, backtest_preds))
    rmse = float(np.sqrt(mean_squared_error(test_values, backtest_preds)))
    r2 = float(r2_score(test_values, backtest_preds))

    # Final forecast: refit on the full series, project `horizon_months` ahead
    final_fit = _fit_sarima(values)
    forecast_result = final_fit.get_forecast(steps=horizon_months)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)

    if np.any(np.isnan(forecast_mean)) or np.any(np.isinf(forecast_mean)):
        raise ValueError("forecast contains NaN/inf values")

    future_dates = pd.date_range(dates.iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")

    forecast_points = []
    for i, fdate in enumerate(future_dates):
        lower, upper = float(conf_int[i][0]), float(conf_int[i][1])
        if upper <= lower:
            raise ValueError(f"non-positive CI width at {fdate}: lower={lower}, upper={upper}")
        forecast_points.append({
            "date": fdate.strftime("%Y-%m-%d"),
            "value": float(forecast_mean[i]),
            "lower": lower,
            "upper": upper,
        })

    return {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "history": [
            {"date": d.strftime("%Y-%m-%d"), "actual": float(v)}
            for d, v in zip(dates, values)
        ],
        "forecast": forecast_points,
        "metrics": {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)},
    }


def update_leaderboard(leaderboard_path: str, live_metrics: dict) -> dict:
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f:
            data = json.load(f)
        models = data["models"]
    else:
        models = copy.deepcopy(FROZEN_MODELS)

    baseline_mae = next(m["mae"] for m in models if m["name"] == "RF Sin TDA")
    for m in models:
        if m["name"] == "SARIMA":
            m["mae"] = round(live_metrics["mae"], 4)
            m["rmse"] = round(live_metrics["rmse"], 4)
            m["r2"] = round(live_metrics["r2"], 4)
            m["improvement_vs_baseline"] = round((baseline_mae - live_metrics["mae"]) / baseline_mae * 100, 2)
            m["live"] = True

    return {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": models,
    }


def main() -> int:
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.abspath(os.path.join(repo_root, "data", "input", "WPUSI01102B.csv"))
    forecast_path = os.path.abspath(os.path.join(repo_root, "dashboard", "public", "data", "forecast.json"))
    leaderboard_path = os.path.abspath(os.path.join(repo_root, "dashboard", "public", "data", "leaderboard.json"))

    try:
        forecast = compute_forecast(csv_path)
        leaderboard = update_leaderboard(leaderboard_path, forecast["metrics"])
    except ValueError as exc:
        print(f"refit_sarima failed sanity check: {exc}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
    with open(forecast_path, "w") as f:
        json.dump(forecast, f, indent=2)
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)

    print(f"Wrote {forecast_path} and {leaderboard_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
