# Leaderboard Fix + Prediction Accuracy Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing LSTM+TDA models to the leaderboard and build a `prediction_track_record.json` archive so the dashboard can eventually show forecast accuracy over time.

**Architecture:** Two independent additions to `scripts/refit_sarima.py`: two new frozen `FROZEN_MODELS` rows, and a new `append_prediction_record()` function called from `main()` that appends the 1-step-ahead forecast to an archive file, idempotent by `target_date`. The GitHub Action commits the new archive file alongside the existing two.

**Tech Stack:** Python (existing `scripts/` toolchain — pandas, pytest), GitHub Actions.

## Global Constraints

- LSTM+TDA rows are frozen (`live: false`) forever — no TensorFlow/Keras in the automated cron.
- `prediction_track_record.json` only stores the 1-step-ahead (`forecast[0]`) prediction per run, not the full 12-month horizon.
- Idempotent by `target_date` — an existing entry is never duplicated or overwritten, even if the workflow reruns within the same month.
- A corrupted/invalid existing archive file must raise (fail loud), never be silently overwritten.
- New frozen model values (verbatim, from the project's notebooks):
  - `LSTM+TDA`: mae=29.4890, rmse=37.5017, r2=0.5687, improvement_vs_baseline=28.41
  - `LSTM+TDA v2+Exog+Attention`: mae=29.1069, rmse=38.6295, r2=0.5424, improvement_vs_baseline=29.34

---

### Task 1: Add LSTM+TDA rows to the leaderboard

**Files:**
- Modify: `scripts/refit_sarima.py:15-20` (the `FROZEN_MODELS` list)
- Modify: `scripts/test_refit_sarima.py` (extend existing leaderboard tests)

**Interfaces:**
- Consumes: nothing new — extends the existing `FROZEN_MODELS` constant and `update_leaderboard()` function, both already defined in `scripts/refit_sarima.py`.
- Produces: `FROZEN_MODELS` now has 6 entries instead of 4. No signature changes to any function.

- [ ] **Step 1: Write the failing test**

Add to `scripts/test_refit_sarima.py` (alongside the existing `test_update_leaderboard_creates_from_frozen_when_missing` test):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest scripts/test_refit_sarima.py -v -k "lstm_tda"`
Expected: FAIL — `AssertionError` on the `"LSTM+TDA" in names` check, since `FROZEN_MODELS` doesn't have these rows yet.

- [ ] **Step 3: Add the two rows to `FROZEN_MODELS`**

In `scripts/refit_sarima.py`, replace the `FROZEN_MODELS` list (lines 15-20):

```python
FROZEN_MODELS = [
    {"name": "RF Sin TDA", "mae": 41.1941, "rmse": 52.1049, "r2": -0.1312, "improvement_vs_baseline": None, "live": False},
    {"name": "RF + TDA", "mae": 41.0917, "rmse": 52.0403, "r2": -0.1284, "improvement_vs_baseline": 0.25, "live": False},
    {"name": "SARIMA", "mae": 29.4225, "rmse": 36.1094, "r2": 0.4567, "improvement_vs_baseline": 28.58, "live": True},
    {"name": "SARIMA+TDA", "mae": 30.5655, "rmse": 36.4237, "r2": 0.4472, "improvement_vs_baseline": 25.80, "live": False},
    {"name": "LSTM+TDA", "mae": 29.4890, "rmse": 37.5017, "r2": 0.5687, "improvement_vs_baseline": 28.41, "live": False},
    {"name": "LSTM+TDA v2+Exog+Attention", "mae": 29.1069, "rmse": 38.6295, "r2": 0.5424, "improvement_vs_baseline": 29.34, "live": False},
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest scripts/test_refit_sarima.py -v`
Expected: all tests PASS, including the 2 new ones (should be 6 total, up from 4).

- [ ] **Step 5: Commit**

```bash
git add scripts/refit_sarima.py scripts/test_refit_sarima.py
git commit -m "feat: add LSTM+TDA models to the leaderboard"
```

---

### Task 2: Prediction accuracy archive

**Files:**
- Modify: `scripts/refit_sarima.py` (add `append_prediction_record()`, call it from `main()`)
- Modify: `scripts/test_refit_sarima.py` (new tests for `append_prediction_record`)

**Interfaces:**
- Consumes: a `forecast_point: dict` with keys `date`, `value`, `lower`, `upper` (the shape of one entry in `compute_forecast()`'s `forecast` list — Task 1 of the original plan already established this).
- Produces: `append_prediction_record(record_path: str, forecast_point: dict, made_on: str) -> list[dict]` — returns the full (possibly unchanged) list of records after the call. Each record has keys `target_date`, `made_on`, `predicted`, `lower`, `upper`.

- [ ] **Step 1: Write the failing test**

Add to `scripts/test_refit_sarima.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest scripts/test_refit_sarima.py -v -k "append_prediction_record"`
Expected: FAIL with `NameError: name 'append_prediction_record' is not defined`.

- [ ] **Step 3: Implement `append_prediction_record`**

Add to `scripts/refit_sarima.py`, after `update_leaderboard` (before `main`):

```python
def append_prediction_record(record_path: str, forecast_point: dict, made_on: str) -> list[dict]:
    if os.path.exists(record_path):
        with open(record_path) as f:
            records = json.load(f)  # raises json.JSONDecodeError on corrupted content, by design
    else:
        records = []

    target_date = forecast_point["date"]
    if any(r["target_date"] == target_date for r in records):
        return records  # idempotent: first prediction for a month wins, reruns don't overwrite

    records.append({
        "target_date": target_date,
        "made_on": made_on,
        "predicted": forecast_point["value"],
        "lower": forecast_point["lower"],
        "upper": forecast_point["upper"],
    })

    with open(record_path, "w") as f:
        json.dump(records, f, indent=2)

    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest scripts/test_refit_sarima.py -v`
Expected: all tests PASS (10 total: 6 from before Task 1/2 plus 4 new).

- [ ] **Step 5: Wire it into `main()`**

In `scripts/refit_sarima.py`, modify `main()`:

```python
def main() -> int:
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.abspath(os.path.join(repo_root, "data", "input", "WPUSI01102B.csv"))
    forecast_path = os.path.abspath(os.path.join(repo_root, "dashboard", "public", "data", "forecast.json"))
    leaderboard_path = os.path.abspath(os.path.join(repo_root, "dashboard", "public", "data", "leaderboard.json"))
    record_path = os.path.abspath(os.path.join(repo_root, "dashboard", "public", "data", "prediction_track_record.json"))

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

    made_on = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    append_prediction_record(record_path, forecast["forecast"][0], made_on=made_on)

    print(f"Wrote {forecast_path}, {leaderboard_path}, and {record_path}")
    return 0
```

Note: `append_prediction_record` runs only after both JSON files have already been written successfully — if `compute_forecast` raised, `main()` already returned 1 before reaching this point, so the archive is never touched on a failed run, matching the fail-loud/no-partial-write convention already established.

- [ ] **Step 6: Run the full test suite**

Run: `pytest scripts/ -v`
Expected: all tests PASS (should be 11 total across all `scripts/test_*.py` files: the 10 in `test_refit_sarima.py` plus 1 unrelated existing test elsewhere — check the actual count matches what was there before this task plus the 4 new ones).

- [ ] **Step 7: Commit**

```bash
git add scripts/refit_sarima.py scripts/test_refit_sarima.py
git commit -m "feat: archive 1-step-ahead predictions for accuracy tracking"
```

---

### Task 3: Wire the archive into the GitHub Action and re-seed real data

**Files:**
- Modify: `.github/workflows/update-forecast.yml:38` (add the new file to the commit step's `git add`)
- Create (generated, not hand-written): `dashboard/public/data/prediction_track_record.json` (seeded with one real entry)
- Modify (generated, not hand-written): `dashboard/public/data/leaderboard.json` (re-seeded with the 6-row leaderboard from Task 1)

**Interfaces:**
- Consumes: `append_prediction_record`, `update_leaderboard`, `compute_forecast`, `FROZEN_MODELS` from Tasks 1-2 (all in `scripts/refit_sarima.py`).
- Produces: real seeded data files the frontend (a later, separate spec) will consume.

- [ ] **Step 1: Update the GitHub Action's commit step**

In `.github/workflows/update-forecast.yml`, change line 38 from:

```yaml
          git add data/input/WPUSI01102B.csv dashboard/public/data/forecast.json dashboard/public/data/leaderboard.json
```

to:

```yaml
          git add data/input/WPUSI01102B.csv dashboard/public/data/forecast.json dashboard/public/data/leaderboard.json dashboard/public/data/prediction_track_record.json
```

- [ ] **Step 2: Verify workflow YAML is still valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/update-forecast.yml')); print('valid yaml')"
```

Expected: `valid yaml`

- [ ] **Step 3: Re-run the SARIMA script against real data to re-seed the leaderboard and create the initial track record**

```bash
source .venv/bin/activate  # or whichever venv has scripts/requirements.txt installed
python scripts/refit_sarima.py
```

Expected: `Wrote .../forecast.json, .../leaderboard.json, and .../prediction_track_record.json`

- [ ] **Step 4: Spot-check the output**

```bash
python3 -c "import json; d=json.load(open('dashboard/public/data/leaderboard.json')); print(sorted((m['name'], m['mae']) for m in d['models']))"
python3 -c "import json; d=json.load(open('dashboard/public/data/prediction_track_record.json')); print(d)"
```

Expected: leaderboard prints 6 models including `LSTM+TDA` and `LSTM+TDA v2+Exog+Attention`; track record prints a list with exactly one entry, `target_date` one month after the latest date in `data/input/WPUSI01102B.csv`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/update-forecast.yml dashboard/public/data/leaderboard.json dashboard/public/data/prediction_track_record.json
git commit -m "chore: wire prediction archive into CI and re-seed leaderboard/track-record data"
```

---

## Post-plan verification

After all 3 tasks: `pytest scripts/ -v` should show all tests passing (11 total). `dashboard/public/data/leaderboard.json` should have 6 models sorted correctly when the frontend renders them (LSTM+TDA v2+Exog+Attention has the lowest MAE). `dashboard/public/data/prediction_track_record.json` should exist with one seeded entry, ready for the follow-up frontend spec (TDA animation + interactivity) to consume.
