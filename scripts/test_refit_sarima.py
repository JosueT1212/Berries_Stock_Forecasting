import json

import numpy as np
import pandas as pd
import pytest

from scripts.refit_sarima import compute_forecast, update_leaderboard, append_prediction_record, FROZEN_MODELS


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


def test_frozen_models_includes_lstm_tda_variants():
    names = {m["name"] for m in FROZEN_MODELS}
    assert "LSTM+TDA" in names
    assert "LSTM+TDA v2+Exog+Attention" in names

    lstm_tda = next(m for m in FROZEN_MODELS if m["name"] == "LSTM+TDA")
    assert lstm_tda["mae"] == 29.4890
    assert lstm_tda["rmse"] == 37.5017
    assert lstm_tda["r2"] == 0.5687
    assert lstm_tda["improvement_vs_baseline"] == 28.41
    assert lstm_tda["live"] is False

    lstm_tda_v2 = next(m for m in FROZEN_MODELS if m["name"] == "LSTM+TDA v2+Exog+Attention")
    assert lstm_tda_v2["mae"] == 29.1069
    assert lstm_tda_v2["rmse"] == 38.6295
    assert lstm_tda_v2["r2"] == 0.5424
    assert lstm_tda_v2["improvement_vs_baseline"] == 29.34
    assert lstm_tda_v2["live"] is False


def test_update_leaderboard_sorts_lstm_tda_v2_as_best_mae():
    # LSTM+TDA v2+Exog+Attention (mae 29.1069) should be the lowest-MAE row
    # once SARIMA's live metrics (mae ~29.42, from the frozen baseline) are applied.
    result = update_leaderboard("/nonexistent/leaderboard.json", {"mae": 29.4225, "rmse": 36.1094, "r2": 0.4567})
    best = min(result["models"], key=lambda m: m["mae"])
    assert best["name"] == "LSTM+TDA v2+Exog+Attention"


def test_append_prediction_record_creates_file_with_one_entry(tmp_path):
    record_path = tmp_path / "prediction_track_record.json"
    forecast_point = {"date": "2026-05-01", "value": 149.1, "lower": 87.2, "upper": 210.9}

    result = append_prediction_record(str(record_path), forecast_point, made_on="2026-04-05")

    assert len(result) == 1
    assert result[0] == {
        "target_date": "2026-05-01",
        "made_on": "2026-04-05",
        "predicted": 149.1,
        "lower": 87.2,
        "upper": 210.9,
    }
    with open(record_path) as f:
        assert json.load(f) == result


def test_append_prediction_record_is_idempotent_by_target_date(tmp_path):
    record_path = tmp_path / "prediction_track_record.json"
    first = {"date": "2026-05-01", "value": 149.1, "lower": 87.2, "upper": 210.9}
    rerun = {"date": "2026-05-01", "value": 999.0, "lower": 1.0, "upper": 2.0}

    append_prediction_record(str(record_path), first, made_on="2026-04-05")
    result = append_prediction_record(str(record_path), rerun, made_on="2026-04-06")

    assert len(result) == 1
    assert result[0]["predicted"] == 149.1  # first value wins, rerun is ignored


def test_append_prediction_record_appends_new_target_date(tmp_path):
    record_path = tmp_path / "prediction_track_record.json"
    may = {"date": "2026-05-01", "value": 149.1, "lower": 87.2, "upper": 210.9}
    june = {"date": "2026-06-01", "value": 155.0, "lower": 90.0, "upper": 220.0}

    append_prediction_record(str(record_path), may, made_on="2026-04-05")
    result = append_prediction_record(str(record_path), june, made_on="2026-05-05")

    assert len(result) == 2
    assert {r["target_date"] for r in result} == {"2026-05-01", "2026-06-01"}
    assert next(r for r in result if r["target_date"] == "2026-05-01")["predicted"] == 149.1


def test_append_prediction_record_raises_on_corrupted_file(tmp_path):
    record_path = tmp_path / "prediction_track_record.json"
    record_path.write_text("not valid json{{{")
    forecast_point = {"date": "2026-05-01", "value": 149.1, "lower": 87.2, "upper": 210.9}

    with pytest.raises(json.JSONDecodeError):
        append_prediction_record(str(record_path), forecast_point, made_on="2026-04-05")
