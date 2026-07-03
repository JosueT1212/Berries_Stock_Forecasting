# Live Forecast Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live, auto-updating React+Vite dashboard (deployed on Vercel) that shows the Berries PPI forecast, model leaderboard, and TDA analysis — replacing "open a notebook" with "look at a URL."

**Architecture:** A monthly GitHub Action pulls the newest FRED data point, refits SARIMA, and writes static JSON into `dashboard/public/data/`. The React SPA fetches that JSON client-side and renders it with recharts (2D) and react-three-fiber (3D embedding). TDA persistence diagrams and subperiod comparisons are computed once from the existing notebook pipeline and committed as static JSON — they are not part of the monthly cron.

**Tech Stack:** Python 3 (pandas, statsmodels, requests, scikit-learn, giotto-tda/ripser for the one-off TDA export, pytest) · React + Vite + TypeScript · Tailwind CSS · recharts · @react-three/fiber + @react-three/drei · GitHub Actions · Vercel

## Global Constraints

- Source of truth for raw data stays `data/input/WPUSI01102B.csv` (existing file, two columns: `observation_date`, `WPUSI01102B`).
- SARIMA order for the live model: `order=(1,0,1)`, `seasonal_order=(1,0,1,12)`, `trend='c'`, `enforce_stationarity=False`, `enforce_invertibility=False` — matches the values already in `data/output/comparacion_modelos_2025_2026.csv` (MAE=29.4225, RMSE=36.1094, R²=0.4567), so the live leaderboard row starts consistent with the notebook's frozen rows.
- Takens embedding params for TDA export: `dimension=12`, `time_delay=3` (per README/CLAUDE.md core pattern).
- Subperiods: `2008-06-01`–`2012-12-31`, `2013-01-01`–`2019-12-31`, `2020-01-01`–`2026-04-30` (per `CLAUDE.md`).
- No TDA or LSTM retraining in the automated cron — only SARIMA is refit monthly.
- All new Python scripts live in `scripts/`, all new frontend code in `dashboard/`.

---

### Task 1: Python scripts environment

**Files:**
- Create: `scripts/requirements.txt`
- Create: `scripts/requirements-tda.txt`
- Create: `scripts/__init__.py` (empty, makes `scripts` importable for tests)
- Test: none (setup task, verified by Task 2's test importing successfully)

**Interfaces:**
- Produces: a `.venv` at repo root that Tasks 2–4 assume is active, with `scripts/requirements.txt` installed.

- [ ] **Step 1: Create requirements files**

`scripts/requirements.txt`:
```
pandas==2.2.3
numpy==1.26.4
statsmodels==0.14.4
requests==2.32.3
scikit-learn==1.5.2
pytest==8.3.3
```

`scripts/requirements-tda.txt`:
```
-r requirements.txt
giotto-tda==0.6.0
ripser==0.6.10
persim==0.3.5
```

- [ ] **Step 2: Create venv and install base requirements**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

Expected: installs cleanly, `python -c "import statsmodels, pandas, requests, sklearn, pytest"` exits 0.

- [ ] **Step 3: Create `scripts/__init__.py`**

Empty file.

- [ ] **Step 4: Commit**

```bash
git add scripts/requirements.txt scripts/requirements-tda.txt scripts/__init__.py
git commit -m "chore: add python scripts environment for dashboard data pipeline"
```

---

### Task 2: `fetch_fred.py` — pull new FRED observations

**Files:**
- Create: `scripts/fetch_fred.py`
- Test: `scripts/test_fetch_fred.py`

**Interfaces:**
- Produces: `fetch_new_observations(csv_path: str, api_key: str, series_id: str = "WPUSI01102B") -> int` — returns count of new rows appended. Appends rows to the CSV at `csv_path` in place, sorted by date, no duplicate dates.
- Produces: `main()` — CLI entry point, reads `FRED_API_KEY` env var and `data/input/WPUSI01102B.csv` path, calls `fetch_new_observations`, prints count, exits 1 if the API call fails.

- [ ] **Step 1: Write the failing test**

`scripts/test_fetch_fred.py`:
```python
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.fetch_fred import fetch_new_observations


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["observation_date", "WPUSI01102B"])
        writer.writerows(rows)


def _read_csv(path):
    with open(path) as f:
        return list(csv.reader(f))


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_appends_only_new_observations(tmp_path):
    csv_path = tmp_path / "WPUSI01102B.csv"
    _write_csv(csv_path, [["2026-03-01", "82.509"], ["2026-04-01", "100.844"]])

    fake_payload = {
        "observations": [
            {"date": "2026-04-01", "value": "100.844"},
            {"date": "2026-05-01", "value": "111.222"},
            {"date": "2026-06-01", "value": "."},  # FRED uses "." for missing values
        ]
    }

    with patch("scripts.fetch_fred.requests.get", return_value=FakeResponse(fake_payload)):
        added = fetch_new_observations(str(csv_path), api_key="fake-key")

    assert added == 1  # only 2026-05-01 is new; 2026-04-01 already present, 2026-06-01 is missing
    rows = _read_csv(csv_path)
    assert rows[-1] == ["2026-05-01", "111.222"]
    assert rows[-2] == ["2026-04-01", "100.844"]


def test_no_new_observations_returns_zero(tmp_path):
    csv_path = tmp_path / "WPUSI01102B.csv"
    _write_csv(csv_path, [["2026-04-01", "100.844"]])

    fake_payload = {"observations": [{"date": "2026-04-01", "value": "100.844"}]}

    with patch("scripts.fetch_fred.requests.get", return_value=FakeResponse(fake_payload)):
        added = fetch_new_observations(str(csv_path), api_key="fake-key")

    assert added == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest scripts/test_fetch_fred.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.fetch_fred'`

- [ ] **Step 3: Write implementation**

`scripts/fetch_fred.py`:
```python
import csv
import os
import sys

import requests

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_new_observations(csv_path: str, api_key: str, series_id: str = "WPUSI01102B") -> int:
    """Fetch observations from FRED and append any not already present in csv_path.

    Returns the number of new rows appended. Rows with a missing value
    (FRED represents these as ".") are skipped.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, existing_rows = rows[0], rows[1:]
    existing_dates = {row[0] for row in existing_rows}

    response = requests.get(
        FRED_BASE_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        },
        timeout=30,
    )
    response.raise_for_status()
    observations = response.json()["observations"]

    new_rows = []
    for obs in observations:
        if obs["date"] in existing_dates:
            continue
        if obs["value"] == ".":
            continue
        new_rows.append([obs["date"], obs["value"]])
        existing_dates.add(obs["date"])

    if new_rows:
        all_rows = existing_rows + new_rows
        all_rows.sort(key=lambda r: r[0])
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_rows)

    return len(new_rows)


def main() -> int:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY environment variable is required", file=sys.stderr)
        return 1

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "input", "WPUSI01102B.csv")
    try:
        added = fetch_new_observations(os.path.abspath(csv_path), api_key)
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        print(f"fetch_fred failed: {exc}", file=sys.stderr)
        return 1

    print(f"Added {added} new observation(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest scripts/test_fetch_fred.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_fred.py scripts/test_fetch_fred.py
git commit -m "feat: add FRED data fetch script"
```

---

### Task 3: `refit_sarima.py` — refit SARIMA, forecast, write JSON

**Files:**
- Create: `scripts/refit_sarima.py`
- Test: `scripts/test_refit_sarima.py`

**Interfaces:**
- Consumes: `data/input/WPUSI01102B.csv` (columns `observation_date`, `WPUSI01102B`).
- Produces: `compute_forecast(csv_path: str, backtest_months: int = 16, horizon_months: int = 12) -> dict` — returns a dict matching the `forecast.json` shape below. Raises `ValueError` if sanity checks fail (NaN/inf in forecast, non-positive CI width).
- Produces: `update_leaderboard(leaderboard_path: str, live_metrics: dict) -> dict` — loads existing `leaderboard.json` (or starts from the frozen baseline rows below if missing), replaces the `"SARIMA"` row's `mae`/`rmse`/`r2`/`improvement_vs_baseline` with `live_metrics`, returns the full updated dict.
- Produces: `main()` — CLI entry point, writes both `dashboard/public/data/forecast.json` and `dashboard/public/data/leaderboard.json`.

`forecast.json` shape:
```json
{
  "last_updated": "2026-07-02T00:00:00Z",
  "history": [{"date": "2008-06-01", "actual": 100.0}],
  "forecast": [{"date": "2026-05-01", "value": 123.4, "lower": 100.1, "upper": 146.7}],
  "metrics": {"mae": 29.4225, "rmse": 36.1094, "r2": 0.4567}
}
```

Frozen baseline rows (used the first time `leaderboard.json` doesn't exist yet):
```python
FROZEN_MODELS = [
    {"name": "RF Sin TDA", "mae": 41.1941, "rmse": 52.1049, "r2": -0.1312, "improvement_vs_baseline": None, "live": False},
    {"name": "RF + TDA", "mae": 41.0917, "rmse": 52.0403, "r2": -0.1284, "improvement_vs_baseline": 0.25, "live": False},
    {"name": "SARIMA", "mae": 29.4225, "rmse": 36.1094, "r2": 0.4567, "improvement_vs_baseline": 28.58, "live": True},
    {"name": "SARIMA+TDA", "mae": 30.5655, "rmse": 36.4237, "r2": 0.4472, "improvement_vs_baseline": 25.80, "live": False},
]
```

- [ ] **Step 1: Write the failing test**

`scripts/test_refit_sarima.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest scripts/test_refit_sarima.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.refit_sarima'`

- [ ] **Step 3: Write implementation**

`scripts/refit_sarima.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest scripts/test_refit_sarima.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/refit_sarima.py scripts/test_refit_sarima.py
git commit -m "feat: add SARIMA refit script producing forecast/leaderboard JSON"
```

---

### Task 4: `export_tda_viz.py` — one-off TDA visualization export

**Files:**
- Create: `scripts/export_tda_viz.py`
- Test: `scripts/test_export_tda_viz.py`

**Interfaces:**
- Consumes: `data/input/WPUSI01102B.csv`.
- Produces: `build_tda_export(csv_path: str) -> dict` — returns a dict matching the `tda.json` shape below.
- Produces: `main()` — CLI entry point, writes `dashboard/public/data/tda.json`. Run manually (`python scripts/export_tda_viz.py`), not part of the cron.

`tda.json` shape:
```json
{
  "generated_at": "2026-07-02T00:00:00Z",
  "subperiods": [
    {
      "id": "2008_2012",
      "label": "2008–2012 (Financial Crisis)",
      "start": "2008-06-01",
      "end": "2012-12-31",
      "n_months": 55,
      "persistence_diagram": [{"birth": 0.12, "death": 0.87, "dim": 0}],
      "max_h1_persistence": 3.42,
      "embedding_3d": [[0.1, -0.2, 0.05]]
    }
  ]
}
```

- [ ] **Step 1: Write the failing test**

`scripts/test_export_tda_viz.py`:
```python
import numpy as np
import pandas as pd

from scripts.export_tda_viz import build_tda_export, SUBPERIODS


def _make_synthetic_csv(path):
    # Cover all three subperiod windows with a seasonal signal so Takens
    # embedding + Vietoris-Rips have real cycles to find.
    dates = pd.date_range("2008-06-01", "2026-04-01", freq="MS")
    t = np.arange(len(dates))
    values = 100 + 15 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(1).normal(0, 3, len(dates))
    df = pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"), "WPUSI01102B": values})
    df.to_csv(path, index=False)


def test_build_tda_export_has_all_subperiods(tmp_path):
    csv_path = tmp_path / "series.csv"
    _make_synthetic_csv(csv_path)

    result = build_tda_export(str(csv_path))

    assert "generated_at" in result
    assert len(result["subperiods"]) == len(SUBPERIODS)
    for sp in result["subperiods"]:
        assert sp["n_months"] > 0
        assert isinstance(sp["max_h1_persistence"], float)
        assert all(len(pt) == 3 for pt in sp["embedding_3d"])
        assert all(set(d.keys()) == {"birth", "death", "dim"} for d in sp["persistence_diagram"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pip install -r scripts/requirements-tda.txt && pytest scripts/test_export_tda_viz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.export_tda_viz'`

- [ ] **Step 3: Write implementation**

`scripts/export_tda_viz.py`:
```python
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SingleTakensEmbedding
from sklearn.decomposition import PCA

EMBEDDING_DIM = 12
TIME_DELAY = 3

SUBPERIODS = [
    {"id": "2008_2012", "label": "2008–2012 (Financial Crisis)", "start": "2008-06-01", "end": "2012-12-31"},
    {"id": "2013_2019", "label": "2013–2019 (Stability)", "start": "2013-01-01", "end": "2019-12-31"},
    {"id": "2020_2026", "label": "2020–2026 (Pandemic/Inflation)", "start": "2020-01-01", "end": "2026-04-30"},
]


def _subperiod_tda(prices: np.ndarray) -> dict:
    ste = SingleTakensEmbedding(parameters_type="fixed", dimension=EMBEDDING_DIM, time_delay=TIME_DELAY)
    embedding = ste.fit_transform(prices)

    vr = VietorisRipsPersistence(homology_dimensions=[0, 1])
    diagram = vr.fit_transform([embedding])[0]  # (n_points, 3): birth, death, homology_dim

    persistence_diagram = [
        {"birth": float(b), "death": float(d), "dim": int(dim)}
        for b, d, dim in diagram
    ]

    h1_mask = diagram[:, 2] == 1
    max_h1 = float(np.max(diagram[h1_mask, 1] - diagram[h1_mask, 0])) if np.any(h1_mask) else 0.0

    n_components = min(3, embedding.shape[0], embedding.shape[1])
    pca = PCA(n_components=n_components)
    embedding_3d = pca.fit_transform(embedding)
    if n_components < 3:
        pad = np.zeros((embedding_3d.shape[0], 3 - n_components))
        embedding_3d = np.hstack([embedding_3d, pad])

    return {
        "persistence_diagram": persistence_diagram,
        "max_h1_persistence": max_h1,
        "embedding_3d": embedding_3d.tolist(),
    }


def build_tda_export(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)
    dates = df["observation_date"].values
    prices = df["WPUSI01102B"].values.astype(float)

    subperiods_out = []
    for sp in SUBPERIODS:
        mask = (dates >= np.datetime64(sp["start"])) & (dates <= np.datetime64(sp["end"]))
        sp_prices = prices[mask]
        tda = _subperiod_tda(sp_prices)
        subperiods_out.append({
            "id": sp["id"],
            "label": sp["label"],
            "start": sp["start"],
            "end": sp["end"],
            "n_months": int(mask.sum()),
            **tda,
        })

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subperiods": subperiods_out,
    }


def main() -> int:
    import os

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.abspath(os.path.join(repo_root, "data", "input", "WPUSI01102B.csv"))
    out_path = os.path.abspath(os.path.join(repo_root, "dashboard", "public", "data", "tda.json"))

    result = build_tda_export(csv_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest scripts/test_export_tda_viz.py -v`
Expected: PASS (1 test). Note: `giotto-tda` install can take a few minutes (compiles native deps) — this is expected.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_tda_viz.py scripts/test_export_tda_viz.py
git commit -m "feat: add one-off TDA visualization export script"
```

---

### Task 5: Seed real dashboard data

**Files:**
- Create (generated, not hand-written): `dashboard/public/data/forecast.json`, `dashboard/public/data/leaderboard.json`, `dashboard/public/data/tda.json`

**Interfaces:**
- Consumes: `compute_forecast`, `update_leaderboard` from Task 3; `build_tda_export` from Task 4.
- Produces: real JSON files the frontend (Tasks 8+) will fetch during local dev, before the dashboard is deployed and the cron has run for the first time.

- [ ] **Step 1: Run the SARIMA script against real data**

```bash
source .venv/bin/activate
mkdir -p dashboard/public/data
python scripts/refit_sarima.py
```

Expected: `Wrote .../dashboard/public/data/forecast.json and .../leaderboard.json`

- [ ] **Step 2: Run the TDA export script against real data**

```bash
pip install -r scripts/requirements-tda.txt
python scripts/export_tda_viz.py
```

Expected: `Wrote .../dashboard/public/data/tda.json`

- [ ] **Step 3: Spot-check the output**

```bash
python -c "import json; d=json.load(open('dashboard/public/data/forecast.json')); print(d['metrics']); print(len(d['forecast']), 'forecast points'); print(len(d['history']), 'history points')"
python -c "import json; d=json.load(open('dashboard/public/data/leaderboard.json')); print([m['name'] for m in d['models']])"
python -c "import json; d=json.load(open('dashboard/public/data/tda.json')); print([s['id'] for s in d['subperiods']])"
```

Expected: metrics roughly matching `data/output/comparacion_modelos_2025_2026.csv` SARIMA row, 12 forecast points, 216 history points, 4 leaderboard models, 3 TDA subperiods.

- [ ] **Step 4: Commit**

```bash
git add dashboard/public/data/forecast.json dashboard/public/data/leaderboard.json dashboard/public/data/tda.json
git commit -m "chore: seed dashboard with initial forecast/leaderboard/tda data"
```

---

### Task 6: GitHub Action for monthly refresh

**Files:**
- Create: `.github/workflows/update-forecast.yml`
- Modify: `README.md` (append a "Live Dashboard" section documenting the `FRED_API_KEY` secret requirement)

**Interfaces:**
- Consumes: `scripts/fetch_fred.py` (`main()`), `scripts/refit_sarima.py` (`main()`).
- Produces: a scheduled workflow that keeps `data/input/WPUSI01102B.csv` and `dashboard/public/data/{forecast,leaderboard}.json` current.

- [ ] **Step 1: Write the workflow**

`.github/workflows/update-forecast.yml`:
```yaml
name: Update Forecast

on:
  schedule:
    - cron: "0 12 5 * *"  # 12:00 UTC on the 5th of every month
  workflow_dispatch: {}

jobs:
  update:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -r scripts/requirements.txt

      - name: Fetch latest FRED data
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
        run: python scripts/fetch_fred.py

      - name: Refit SARIMA and write dashboard data
        run: python scripts/refit_sarima.py

      - name: Commit changes if any
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/input/WPUSI01102B.csv dashboard/public/data/forecast.json dashboard/public/data/leaderboard.json
          git diff --cached --quiet || git commit -m "chore: monthly forecast update"
          git push
```

- [ ] **Step 2: Verify workflow syntax**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/update-forecast.yml'))" 2>/dev/null || pip install pyyaml && python -c "import yaml; yaml.safe_load(open('.github/workflows/update-forecast.yml')); print('valid yaml')"
```

Expected: `valid yaml`, no exception.

- [ ] **Step 3: Document the required secret in README**

Append to `README.md`:
```markdown

## Live Dashboard

A React dashboard (`dashboard/`) is deployed on Vercel and auto-updates monthly via
`.github/workflows/update-forecast.yml`, which pulls the newest FRED observation and
refits the SARIMA model.

**Setup:**
1. Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
2. Add it as a GitHub Actions secret named `FRED_API_KEY` (repo Settings → Secrets and variables → Actions).
3. Link `dashboard/` as the root directory of a Vercel project for deployment.

TDA visualizations (`dashboard/public/data/tda.json`) are regenerated manually with
`python scripts/export_tda_viz.py` when the underlying analysis changes — they are not
part of the monthly cron.
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/update-forecast.yml README.md
git commit -m "feat: add monthly forecast-refresh GitHub Action"
```

---

### Task 7: Vite + React + Tailwind scaffold

**Files:**
- Create: `dashboard/package.json`, `dashboard/vite.config.ts`, `dashboard/tsconfig.json`, `dashboard/tsconfig.node.json`, `dashboard/index.html`, `dashboard/tailwind.config.js`, `dashboard/postcss.config.js`, `dashboard/src/main.tsx`, `dashboard/src/index.css`, `dashboard/src/App.tsx`

**Interfaces:**
- Produces: a running Vite dev server at `dashboard/` with Tailwind wired up, ready for Tasks 8–13 to add components into `dashboard/src/`.

- [ ] **Step 1: Scaffold with Vite's official template**

```bash
npm create vite@latest dashboard -- --template react-ts
cd dashboard
npm install
npm install recharts @react-three/fiber @react-three/drei three framer-motion
npm install -D tailwindcss postcss autoprefixer @types/three
npx tailwindcss init -p
cd ..
```

- [ ] **Step 2: Configure Tailwind content paths**

`dashboard/tailwind.config.js`:
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: "#0b0e14", light: "#f7f8fa" },
        accent: { DEFAULT: "#6366f1", warm: "#f59e0b" },
      },
    },
  },
  darkMode: "media",
  plugins: [],
};
```

- [ ] **Step 3: Replace generated `src/index.css` with Tailwind directives + base theme**

`dashboard/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  color-scheme: light dark;
}

body {
  @apply bg-white text-gray-900 dark:bg-bg dark:text-gray-100;
  font-family: "Inter", system-ui, sans-serif;
}
```

- [ ] **Step 4: Replace generated `src/App.tsx` with a placeholder shell**

`dashboard/src/App.tsx`:
```tsx
export default function App() {
  return (
    <div className="min-h-screen p-8">
      <h1 className="text-3xl font-bold">Berries PPI Forecast Dashboard</h1>
      <p className="text-gray-500 dark:text-gray-400">Loading...</p>
    </div>
  );
}
```

- [ ] **Step 5: Verify dev server runs**

```bash
cd dashboard && npm run dev -- --port 5173 &
sleep 3
curl -sf http://localhost:5173 | grep -q "Berries PPI" && echo "OK"
kill %1
cd ..
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add dashboard/package.json dashboard/package-lock.json dashboard/vite.config.ts dashboard/tsconfig*.json dashboard/index.html dashboard/tailwind.config.js dashboard/postcss.config.js dashboard/src/main.tsx dashboard/src/index.css dashboard/src/App.tsx dashboard/.gitignore
git commit -m "chore: scaffold Vite + React + Tailwind dashboard app"
```

---

### Task 8: Data types and loaders

**Files:**
- Create: `dashboard/src/lib/types.ts`
- Create: `dashboard/src/lib/data.ts`
- Test: `dashboard/src/lib/data.test.ts`
- Modify: `dashboard/package.json` (add `vitest`), `dashboard/vite.config.ts` (add test config)

**Interfaces:**
- Produces: `ForecastData`, `LeaderboardData`, `TdaData` TypeScript types matching the JSON shapes from Tasks 3–4.
- Produces: `loadForecast(): Promise<ForecastData>`, `loadLeaderboard(): Promise<LeaderboardData>`, `loadTda(): Promise<TdaData>` — each does `fetch('/data/<name>.json')` and returns parsed JSON, throwing on non-OK response.

- [ ] **Step 1: Install vitest**

```bash
cd dashboard && npm install -D vitest && cd ..
```

- [ ] **Step 2: Add test script and vitest config**

Add to `dashboard/package.json` `"scripts"`:
```json
"test": "vitest run"
```

Append to `dashboard/vite.config.ts`:
```ts
/// <reference types="vitest/config" />
```
and add a `test` block to the exported config object:
```ts
test: {
  environment: "jsdom",
},
```
(also `npm install -D jsdom` in `dashboard/`)

- [ ] **Step 3: Write the failing test**

`dashboard/src/lib/data.test.ts`:
```ts
import { describe, expect, it, vi, beforeEach } from "vitest";
import { loadForecast, loadLeaderboard, loadTda } from "./data";

function mockFetchOnce(payload: unknown, ok = true) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok,
      json: () => Promise.resolve(payload),
    })
  );
}

describe("data loaders", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("loadForecast fetches and parses forecast.json", async () => {
    const payload = { last_updated: "x", history: [], forecast: [], metrics: { mae: 1, rmse: 2, r2: 0.5 } };
    mockFetchOnce(payload);
    const result = await loadForecast();
    expect(result.metrics.r2).toBe(0.5);
    expect(fetch).toHaveBeenCalledWith("/data/forecast.json");
  });

  it("loadLeaderboard throws on non-ok response", async () => {
    mockFetchOnce({}, false);
    await expect(loadLeaderboard()).rejects.toThrow();
  });

  it("loadTda fetches and parses tda.json", async () => {
    const payload = { generated_at: "x", subperiods: [] };
    mockFetchOnce(payload);
    const result = await loadTda();
    expect(result.subperiods).toEqual([]);
  });
});
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd dashboard && npm run test`
Expected: FAIL — `./data` module not found

- [ ] **Step 5: Write types**

`dashboard/src/lib/types.ts`:
```ts
export interface ForecastPoint {
  date: string;
  actual?: number;
  value?: number;
  lower?: number;
  upper?: number;
}

export interface ForecastData {
  last_updated: string;
  history: { date: string; actual: number }[];
  forecast: { date: string; value: number; lower: number; upper: number }[];
  metrics: { mae: number; rmse: number; r2: number };
}

export interface LeaderboardModel {
  name: string;
  mae: number;
  rmse: number;
  r2: number;
  improvement_vs_baseline: number | null;
  live: boolean;
}

export interface LeaderboardData {
  last_updated: string;
  models: LeaderboardModel[];
}

export interface PersistencePoint {
  birth: number;
  death: number;
  dim: number;
}

export interface Subperiod {
  id: string;
  label: string;
  start: string;
  end: string;
  n_months: number;
  persistence_diagram: PersistencePoint[];
  max_h1_persistence: number;
  embedding_3d: [number, number, number][];
}

export interface TdaData {
  generated_at: string;
  subperiods: Subperiod[];
}
```

- [ ] **Step 6: Write loaders**

`dashboard/src/lib/data.ts`:
```ts
import type { ForecastData, LeaderboardData, TdaData } from "./types";

async function loadJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function loadForecast(): Promise<ForecastData> {
  return loadJson<ForecastData>("/data/forecast.json");
}

export function loadLeaderboard(): Promise<LeaderboardData> {
  return loadJson<LeaderboardData>("/data/leaderboard.json");
}

export function loadTda(): Promise<TdaData> {
  return loadJson<TdaData>("/data/tda.json");
}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd dashboard && npm run test`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add dashboard/src/lib dashboard/package.json dashboard/package-lock.json dashboard/vite.config.ts
git commit -m "feat: add typed data loaders for dashboard JSON"
```

---

### Task 9: Forecast chart + stat tiles

**Files:**
- Create: `dashboard/src/components/StatTile.tsx`
- Create: `dashboard/src/components/ForecastChart.tsx`

**Interfaces:**
- Consumes: `ForecastData` from `dashboard/src/lib/types.ts` (Task 8).
- Produces: `<StatTile label={string} value={string} sublabel?={string} />`, `<ForecastChart data={ForecastData} />` — self-contained components consumed by `App.tsx` in Task 13.

- [ ] **Step 1: Write `StatTile`**

`dashboard/src/components/StatTile.tsx`:
```tsx
interface StatTileProps {
  label: string;
  value: string;
  sublabel?: string;
}

export default function StatTile({ label, value, sublabel }: StatTileProps) {
  return (
    <div className="rounded-2xl bg-gray-50 dark:bg-white/5 p-5 flex flex-col gap-1 shadow-sm">
      <span className="text-sm uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</span>
      <span className="text-3xl font-semibold text-accent">{value}</span>
      {sublabel && <span className="text-xs text-gray-400">{sublabel}</span>}
    </div>
  );
}
```

- [ ] **Step 2: Write `ForecastChart`**

Uses recharts' stacked-area trick for the confidence band: plot `lower` as an invisible base area, then `upper - lower` as the visible band on top.

`dashboard/src/components/ForecastChart.tsx`:
```tsx
import {
  Area,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import type { ForecastData } from "../lib/types";

interface ChartRow {
  date: string;
  actual?: number;
  forecast?: number;
  ciBase?: number;
  ciBand?: number;
}

function buildRows(data: ForecastData): ChartRow[] {
  const historyRows: ChartRow[] = data.history.map((h) => ({ date: h.date, actual: h.actual }));
  const forecastRows: ChartRow[] = data.forecast.map((f) => ({
    date: f.date,
    forecast: f.value,
    ciBase: f.lower,
    ciBand: f.upper - f.lower,
  }));
  return [...historyRows.slice(-36), ...forecastRows];
}

export default function ForecastChart({ data }: { data: ForecastData }) {
  const rows = buildRows(data);

  return (
    <div className="h-96 w-full rounded-2xl bg-gray-50 dark:bg-white/5 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={30} />
          <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ borderRadius: 12, border: "none", boxShadow: "0 4px 20px rgba(0,0,0,0.15)" }}
          />
          <Area type="monotone" dataKey="ciBase" stackId="ci" stroke="none" fill="transparent" />
          <Area
            type="monotone"
            dataKey="ciBand"
            stackId="ci"
            stroke="none"
            fill="#6366f1"
            fillOpacity={0.15}
            name="95% CI"
          />
          <Line type="monotone" dataKey="actual" stroke="#0ea5e9" strokeWidth={2} dot={false} name="Actual" />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#f59e0b"
            strokeWidth={2}
            strokeDasharray="6 4"
            dot={{ r: 3 }}
            name="SARIMA Forecast"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 3: Verify it compiles**

```bash
cd dashboard && npx tsc --noEmit
```

Expected: no errors (component not wired into `App.tsx` yet, but must type-check standalone — `tsc --noEmit` checks all files under `src/` regardless of usage).

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/components/StatTile.tsx dashboard/src/components/ForecastChart.tsx
git commit -m "feat: add forecast chart and stat tile components"
```

---

### Task 10: Model leaderboard component

**Files:**
- Create: `dashboard/src/components/Leaderboard.tsx`

**Interfaces:**
- Consumes: `LeaderboardData` from `dashboard/src/lib/types.ts`.
- Produces: `<Leaderboard data={LeaderboardData} />`.

- [ ] **Step 1: Write `Leaderboard`**

`dashboard/src/components/Leaderboard.tsx`:
```tsx
import type { LeaderboardData } from "../lib/types";

export default function Leaderboard({ data }: { data: LeaderboardData }) {
  const sorted = [...data.models].sort((a, b) => a.mae - b.mae);

  return (
    <div className="rounded-2xl bg-gray-50 dark:bg-white/5 p-4 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-white/10">
            <th className="py-2 pr-4">Model</th>
            <th className="py-2 pr-4">MAE</th>
            <th className="py-2 pr-4">RMSE</th>
            <th className="py-2 pr-4">R²</th>
            <th className="py-2 pr-4">Improvement vs. RF Base</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m) => (
            <tr
              key={m.name}
              className={`border-b border-gray-100 dark:border-white/5 ${
                m.live ? "bg-accent/10 font-medium" : ""
              }`}
            >
              <td className="py-2 pr-4 flex items-center gap-2">
                {m.name}
                {m.live && (
                  <span className="text-xs uppercase tracking-wide bg-accent text-white rounded-full px-2 py-0.5">
                    live
                  </span>
                )}
              </td>
              <td className="py-2 pr-4">{m.mae.toFixed(2)}</td>
              <td className="py-2 pr-4">{m.rmse.toFixed(2)}</td>
              <td className="py-2 pr-4">{m.r2.toFixed(3)}</td>
              <td className="py-2 pr-4">
                {m.improvement_vs_baseline === null ? "—" : `${m.improvement_vs_baseline > 0 ? "+" : ""}${m.improvement_vs_baseline.toFixed(2)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-400 mt-2">
        Rows marked <span className="font-medium">live</span> refit monthly. Others are frozen at their notebook-run values.
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd dashboard && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/components/Leaderboard.tsx
git commit -m "feat: add model leaderboard component"
```

---

### Task 11: TDA Explorer component (persistence diagram + 3D embedding)

**Files:**
- Create: `dashboard/src/components/PersistenceDiagram.tsx`
- Create: `dashboard/src/components/EmbeddingScene.tsx`
- Create: `dashboard/src/components/TdaExplorer.tsx`

**Interfaces:**
- Consumes: `Subperiod` from `dashboard/src/lib/types.ts`.
- Produces: `<TdaExplorer data={TdaData} />`, composing `<PersistenceDiagram subperiod={Subperiod} />` (2D scatter, recharts) and `<EmbeddingScene subperiod={Subperiod} />` (3D scatter, react-three-fiber). Includes a subperiod selector (tabs).

- [ ] **Step 1: Write `PersistenceDiagram`**

`dashboard/src/components/PersistenceDiagram.tsx`:
```tsx
import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import type { Subperiod } from "../lib/types";

export default function PersistenceDiagram({ subperiod }: { subperiod: Subperiod }) {
  const h0 = subperiod.persistence_diagram.filter((p) => p.dim === 0);
  const h1 = subperiod.persistence_diagram.filter((p) => p.dim === 1);
  const maxDeath = Math.max(1, ...subperiod.persistence_diagram.map((p) => p.death));

  return (
    <div className="h-80 w-full rounded-2xl bg-gray-50 dark:bg-white/5 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis type="number" dataKey="birth" name="birth" domain={[0, maxDeath]} tick={{ fontSize: 11 }} />
          <YAxis type="number" dataKey="death" name="death" domain={[0, maxDeath]} tick={{ fontSize: 11 }} />
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: maxDeath, y: maxDeath }]}
            stroke="#94a3b8"
            strokeDasharray="4 4"
          />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter name="H₀ (components)" data={h0} fill="#0ea5e9" />
          <Scatter name="H₁ (loops)" data={h1} fill="#f59e0b" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 2: Write `EmbeddingScene`**

`dashboard/src/components/EmbeddingScene.tsx`:
```tsx
import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import type { Subperiod } from "../lib/types";

function PointCloud({ points }: { points: [number, number, number][] }) {
  return (
    <>
      {points.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[0.04, 8, 8]} />
          <meshStandardMaterial color="#6366f1" />
        </mesh>
      ))}
    </>
  );
}

export default function EmbeddingScene({ subperiod }: { subperiod: Subperiod }) {
  return (
    <div className="h-80 w-full rounded-2xl bg-gray-50 dark:bg-white/5 overflow-hidden">
      <Canvas camera={{ position: [3, 3, 3] }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[5, 5, 5]} intensity={1} />
        <PointCloud points={subperiod.embedding_3d} />
        <OrbitControls autoRotate autoRotateSpeed={0.6} enableZoom />
      </Canvas>
    </div>
  );
}
```

- [ ] **Step 3: Write `TdaExplorer`**

`dashboard/src/components/TdaExplorer.tsx`:
```tsx
import { useState } from "react";
import type { TdaData } from "../lib/types";
import PersistenceDiagram from "./PersistenceDiagram";
import EmbeddingScene from "./EmbeddingScene";

export default function TdaExplorer({ data }: { data: TdaData }) {
  const [activeId, setActiveId] = useState(data.subperiods[0]?.id);
  const active = data.subperiods.find((s) => s.id === activeId) ?? data.subperiods[0];

  if (!active) return null;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-2">
        {data.subperiods.map((s) => (
          <button
            key={s.id}
            onClick={() => setActiveId(s.id)}
            className={`px-4 py-2 rounded-full text-sm transition-colors ${
              s.id === active.id
                ? "bg-accent text-white"
                : "bg-gray-100 dark:bg-white/5 text-gray-600 dark:text-gray-300"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PersistenceDiagram subperiod={active} />
        <EmbeddingScene subperiod={active} />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify it compiles**

```bash
cd dashboard && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/components/PersistenceDiagram.tsx dashboard/src/components/EmbeddingScene.tsx dashboard/src/components/TdaExplorer.tsx
git commit -m "feat: add TDA explorer with persistence diagram and 3D embedding"
```

---

### Task 12: Subperiod comparison component

**Files:**
- Create: `dashboard/src/components/SubperiodComparison.tsx`

**Interfaces:**
- Consumes: `TdaData` from `dashboard/src/lib/types.ts`.
- Produces: `<SubperiodComparison data={TdaData} />`.

- [ ] **Step 1: Write `SubperiodComparison`**

`dashboard/src/components/SubperiodComparison.tsx`:
```tsx
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TdaData } from "../lib/types";

const COLORS = ["#0ea5e9", "#6366f1", "#f59e0b"];

export default function SubperiodComparison({ data }: { data: TdaData }) {
  const rows = data.subperiods.map((s) => ({
    label: s.label,
    max_h1_persistence: s.max_h1_persistence,
  }));

  return (
    <div className="rounded-2xl bg-gray-50 dark:bg-white/5 p-4">
      <h3 className="text-sm uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2">
        H₁ Persistence by Subperiod
      </h3>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
            <XAxis dataKey="label" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="max_h1_persistence" radius={[8, 8, 0, 0]}>
              {rows.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-gray-400 mt-2">
        Elevated H₁ persistence in 2008–2012 confirms stronger nonlinear cyclicity during the financial crisis.
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd dashboard && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/components/SubperiodComparison.tsx
git commit -m "feat: add subperiod H1 persistence comparison component"
```

---

### Task 13: App shell — nav, data fetching, assembly

**Files:**
- Modify: `dashboard/src/App.tsx`

**Interfaces:**
- Consumes: `loadForecast`, `loadLeaderboard`, `loadTda` (Task 8); `ForecastChart`, `StatTile` (Task 9); `Leaderboard` (Task 10); `TdaExplorer` (Task 11); `SubperiodComparison` (Task 12).
- Produces: the assembled dashboard page with section tabs (Forecast / Leaderboard / TDA Explorer / Subperiods), loading and error states.

- [ ] **Step 1: Replace `App.tsx` with the full assembly**

`dashboard/src/App.tsx`:
```tsx
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { loadForecast, loadLeaderboard, loadTda } from "./lib/data";
import type { ForecastData, LeaderboardData, TdaData } from "./lib/types";
import StatTile from "./components/StatTile";
import ForecastChart from "./components/ForecastChart";
import Leaderboard from "./components/Leaderboard";
import TdaExplorer from "./components/TdaExplorer";
import SubperiodComparison from "./components/SubperiodComparison";

type Section = "forecast" | "leaderboard" | "tda" | "subperiods";

const SECTIONS: { id: Section; label: string }[] = [
  { id: "forecast", label: "Forecast" },
  { id: "leaderboard", label: "Model Leaderboard" },
  { id: "tda", label: "TDA Explorer" },
  { id: "subperiods", label: "Subperiods" },
];

function isStale(lastUpdated: string): boolean {
  const days = (Date.now() - new Date(lastUpdated).getTime()) / (1000 * 60 * 60 * 24);
  return days > 40;
}

export default function App() {
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardData | null>(null);
  const [tda, setTda] = useState<TdaData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [section, setSection] = useState<Section>("forecast");

  useEffect(() => {
    Promise.all([loadForecast(), loadLeaderboard(), loadTda()])
      .then(([f, l, t]) => {
        setForecast(f);
        setLeaderboard(l);
        setTda(t);
      })
      .catch((err) => setError(String(err)));
  }, []);

  if (error) {
    return <div className="p-8 text-red-500">Failed to load dashboard data: {error}</div>;
  }

  if (!forecast || !leaderboard || !tda) {
    return <div className="p-8 text-gray-400">Loading dashboard...</div>;
  }

  const latest = forecast.history[forecast.history.length - 1];
  const next = forecast.forecast[0];

  return (
    <div className="min-h-screen max-w-6xl mx-auto p-6 md:p-10 flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-3xl font-bold">Berries PPI Forecast Dashboard</h1>
        <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
          <span>Last updated {new Date(forecast.last_updated).toLocaleDateString()}</span>
          {isStale(forecast.last_updated) && (
            <span className="bg-amber-500/20 text-amber-600 dark:text-amber-400 px-2 py-0.5 rounded-full text-xs">
              stale — monthly refresh may have missed
            </span>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatTile label="Latest Actual" value={latest.actual.toFixed(1)} sublabel={latest.date} />
        <StatTile label="Next Month Forecast" value={next.value.toFixed(1)} sublabel={next.date} />
        <StatTile label="Live Model MAE" value={forecast.metrics.mae.toFixed(2)} sublabel="SARIMA, backtest" />
      </div>

      <nav className="flex gap-2 border-b border-gray-200 dark:border-white/10 pb-2">
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => setSection(s.id)}
            className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-colors ${
              section === s.id
                ? "text-accent border-b-2 border-accent"
                : "text-gray-500 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
            }`}
          >
            {s.label}
          </button>
        ))}
      </nav>

      <motion.main
        key={section}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
      >
        {section === "forecast" && <ForecastChart data={forecast} />}
        {section === "leaderboard" && <Leaderboard data={leaderboard} />}
        {section === "tda" && <TdaExplorer data={tda} />}
        {section === "subperiods" && <SubperiodComparison data={tda} />}
      </motion.main>
    </div>
  );
}
```

- [ ] **Step 2: Copy seeded data into dashboard public dir for local dev (already done in Task 5, verify present)**

```bash
ls dashboard/public/data/
```

Expected: `forecast.json  leaderboard.json  tda.json`

- [ ] **Step 3: Run dev server and manually verify all four sections render**

```bash
cd dashboard && npm run dev -- --port 5173 &
sleep 3
curl -sf http://localhost:5173 | grep -q "Berries PPI Forecast Dashboard" && echo "OK: page loads"
kill %1
cd ..
```

Expected: `OK: page loads`. (Full visual check of charts requires opening the browser — do this manually and confirm forecast line, leaderboard table, persistence diagram, 3D point cloud, and subperiod bars all render without console errors.)

- [ ] **Step 4: Type-check and build**

```bash
cd dashboard && npx tsc --noEmit && npm run build
```

Expected: no type errors, build succeeds, `dashboard/dist/` produced.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/App.tsx
git commit -m "feat: assemble dashboard app shell with section navigation"
```

---

### Task 14: Vercel deploy config

**Files:**
- Create: `dashboard/vercel.json`
- Modify: `README.md` (already has the "Live Dashboard" section from Task 6 — add the deployed URL placeholder note)

**Interfaces:**
- Produces: a `vercel.json` so Vercel serves the SPA correctly (rewrites all routes to `index.html` — needed even for a single-page app so a hard refresh doesn't 404).

- [ ] **Step 1: Write `vercel.json`**

`dashboard/vercel.json`:
```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

- [ ] **Step 2: Verify build output matches what Vercel expects**

```bash
cd dashboard && npm run build && ls dist/index.html && cd ..
```

Expected: `dist/index.html` exists (Vite's default build output dir, which is also Vercel's default expectation for a Vite project).

- [ ] **Step 3: Commit**

```bash
git add dashboard/vercel.json
git commit -m "chore: add Vercel SPA rewrite config"
```

- [ ] **Step 4: Deploy (manual, requires user's Vercel account)**

Not automated in this plan — run interactively:
```bash
cd dashboard && npx vercel link && npx vercel --prod
```
Confirm with the user before running `--prod`, since it publishes a public URL.

---

## Post-plan verification

After all 14 tasks: full pytest suite (`pytest scripts/ -v`) and full frontend suite (`cd dashboard && npm run test && npx tsc --noEmit && npm run build`) should both pass, and a manual browser check of `npm run dev` should show working Forecast / Leaderboard / TDA Explorer / Subperiods tabs with real data from `dashboard/public/data/*.json`.
