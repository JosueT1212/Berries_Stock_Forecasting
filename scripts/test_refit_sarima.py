import json

import numpy as np
import pandas as pd
import pytest

from scripts.refit_sarima import compute_forecast, update_leaderboard, FROZEN_MODELS


def _make_synthetic_csv(path, n_months=60):
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    # seasonal + trend + noise, enough signal for SARIMAX to fit without error
    t = np.arange(n_months)
    values = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(0).normal(0, 2, n_months)
    df = pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"), "WPUSI01102B": values})
    df.to_csv(path, index=False)
    return values


def test_compute_forecast_shape_and_sanity(tmp_path):
    csv_path = tmp_path / "series.csv"
    _make_synthetic_csv(csv_path)

    result = compute_forecast(str(csv_path), backtest_months=12, horizon_months=6)

    assert "last_updated" in result
    assert len(result["forecast"]) == 6
    assert all(not np.isnan(pt["value"]) for pt in result["forecast"])
    assert all(pt["upper"] > pt["lower"] for pt in result["forecast"])
    assert set(result["metrics"].keys()) == {"mae", "rmse", "r2"}
    assert len(result["history"]) == 60


def test_compute_forecast_rejects_too_short_series(tmp_path):
    csv_path = tmp_path / "series.csv"
    _make_synthetic_csv(csv_path, n_months=10)

    with pytest.raises(ValueError, match="at least"):
        compute_forecast(str(csv_path), backtest_months=12, horizon_months=6)


def test_update_leaderboard_creates_from_frozen_when_missing(tmp_path):
    lb_path = tmp_path / "leaderboard.json"
    live_metrics = {"mae": 25.0, "rmse": 30.0, "r2": 0.5}

    result = update_leaderboard(str(lb_path), live_metrics)

    sarima_row = next(m for m in result["models"] if m["name"] == "SARIMA")
    assert sarima_row["mae"] == 25.0
    assert sarima_row["live"] is True
    other_names = {m["name"] for m in result["models"]}
    assert other_names == {m["name"] for m in FROZEN_MODELS}


def test_update_leaderboard_preserves_frozen_rows_on_update(tmp_path):
    lb_path = tmp_path / "leaderboard.json"
    lb_path.write_text(json.dumps({"last_updated": "old", "models": FROZEN_MODELS}))

    result = update_leaderboard(str(lb_path), {"mae": 10.0, "rmse": 12.0, "r2": 0.9})

    rf_row = next(m for m in result["models"] if m["name"] == "RF Sin TDA")
    assert rf_row["mae"] == 41.1941  # untouched
    sarima_row = next(m for m in result["models"] if m["name"] == "SARIMA")
    assert sarima_row["mae"] == 10.0
