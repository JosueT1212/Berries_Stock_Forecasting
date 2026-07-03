# Leaderboard Fix + Prediction Accuracy Archive — Design

**Date:** 2026-07-02
**Status:** Approved

## Purpose

Two backend/data gaps found after the live dashboard shipped:

1. The leaderboard is missing the actual best-performing models. `data/output/comparacion_modelos_2025_2026.csv` (and `refit_sarima.py`'s `FROZEN_MODELS`) only cover RF/SARIMA variants — the LSTM+TDA notebooks (Sesion-6, Sesion-8) were never folded in, even though they beat SARIMA on R² and (for Sesion-8) on MAE too.
2. The dashboard only ever shows the *current* forecast — once a month passes, last month's prediction is silently overwritten and there's no way to see how accurate past predictions were.

This spec is backend/data-only. The frontend additions that consume `prediction_track_record.json` (chart markers, accuracy stat tile) are Spec 2's job — this spec just establishes the file and the write-path contract.

## Part 1: Leaderboard fix

Add two frozen rows to `FROZEN_MODELS` in `scripts/refit_sarima.py`, sourced from the notebooks:

```python
{"name": "LSTM+TDA", "mae": 29.4890, "rmse": 37.5017, "r2": 0.5687, "improvement_vs_baseline": None, "live": False},
{"name": "LSTM+TDA v2+Exog+Attention", "mae": 29.1069, "rmse": 38.6295, "r2": 0.5424, "improvement_vs_baseline": None, "live": False},
```

`improvement_vs_baseline` computed the same way as the existing rows: `(baseline_mae - model_mae) / baseline_mae * 100`, rounded to 2 decimals, where baseline is the `RF Sin TDA` row's MAE — this is already how `update_leaderboard()` derives it for the live SARIMA row, so the same formula is applied here at seed time rather than hardcoded. Both new rows stay `live: false` forever — no TensorFlow/Keras in the automated cron (per the original project's non-goals, extended to cover LSTM too).

Re-run `refit_sarima.py`'s leaderboard seed step (or a small one-off) to regenerate `dashboard/public/data/leaderboard.json` with all 6 rows. Sorted by MAE (existing frontend behavior, unchanged), `LSTM+TDA v2+Exog+Attention` becomes the top row; the `live` SARIMA badge stays correctly on the SARIMA row wherever it lands in the sort.

## Part 2: Prediction accuracy archive

**New file:** `dashboard/public/data/prediction_track_record.json`

```json
[
  { "target_date": "2026-05-01", "made_on": "2026-04-05", "predicted": 149.1, "lower": 87.2, "upper": 210.9 }
]
```

- One entry per calendar month that has ever had a next-month forecast made for it. Only the **1-step-ahead** prediction (`forecast[0]` from that run's `compute_forecast()` output) is archived — not the full 12-month horizon. This keeps the record unambiguous: one predicted value per target month, not a tangle of overlapping multi-horizon forecasts.
- `scripts/refit_sarima.py` gains a new function, `append_prediction_record(record_path: str, forecast_point: dict, made_on: str) -> list[dict]`, called from `main()` after `compute_forecast()` succeeds (same pre-write-sanity-check gate as the rest of the pipeline — if `compute_forecast` raised, this never runs).
- **Idempotent by `target_date`:** if an entry for that `target_date` already exists (e.g. the workflow re-runs within the same month), it is **not** duplicated or overwritten — the archive is a permanent historical record, not a live-updating one. The first prediction made for a given month is the one that stays on record, since that's the one being graded for accuracy.
- No backend computation of "was it accurate" — that requires comparing against an actual value that may not exist yet. The frontend (Spec 2) does this join itself: `forecast.json`'s `history[]` already has every actual to date, so any `prediction_track_record.json` entry whose `target_date` is `<=` the latest history date is resolvable client-side, with no backend involvement.

## Error handling & testing

- `append_prediction_record` follows the same fail-loud convention as the rest of `refit_sarima.py`: if `record_path`'s existing content isn't valid JSON, that's a hard error (raises), not a silent overwrite — a corrupted archive should stop the pipeline, not quietly lose history.
- Test: pytest cases for (a) appending a new `target_date` to an empty/missing file creates it with one entry, (b) appending a `target_date` that already exists in the file leaves the file unchanged (idempotency), (c) appending a genuinely new `target_date` to a non-empty file appends without disturbing existing entries.
- The existing GH Action (`update-forecast.yml`) needs one change: `refit_sarima.py`'s `main()` already writes `forecast.json`/`leaderboard.json`; it now also writes `prediction_track_record.json`, and that file needs to be added to the `git add` list in the commit step alongside the other two.

## Non-goals

- No backend-computed accuracy metrics (MAE-over-time, etc.) — that's a frontend concern for Spec 2, computed from data already present in `forecast.json` + `prediction_track_record.json`.
- No retraining or live-refitting of LSTM+TDA models — frozen rows only, per the earlier project non-goal extended to this spec.
- No backfilling of historical predictions before this feature ships — the track record starts accumulating from whenever this ships forward. (There's no way to know what SARIMA would have predicted in past months without re-running history, which is out of scope.)
