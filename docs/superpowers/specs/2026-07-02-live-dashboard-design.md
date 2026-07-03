# Live Forecast Dashboard — Design

**Date:** 2026-07-02
**Status:** Approved

## Purpose

The project currently exists only as Jupyter notebooks, a LaTeX report, and slide decks. Nothing shows the work "live" to someone watching the project (e.g. a course instructor or stakeholder). This spec adds a deployed, auto-updating dashboard that visualizes the forecasting pipeline and its results without requiring anyone to open a notebook.

## Non-goals

- No real-time / sub-monthly data. Source data (FRED `WPUSI01102B`) is a monthly Producer Price Index series — "live" means auto-refreshing on FRED's release cadence, not streaming ticks.
- No retraining of RF / LSTM / TDA-feature models on a schedule. Those stay frozen at their notebook-computed values; only SARIMA (the best-performing model, R²=0.4567) is refit monthly.
- No re-running the full TDA pipeline (Takens embedding + Vietoris-Rips) automatically — it's computed once from notebooks and exported to static JSON, since the topological structure doesn't meaningfully shift month to month.

## Architecture

Monorepo (this repo) gains three new pieces:

- **`dashboard/`** — React + Vite SPA, deployed to Vercel as a static site.
- **`scripts/`** — Python:
  - `fetch_fred.py` — pulls the latest FRED point via the FRED API, appends to `data/input/WPUSI01102B.csv` (source of truth).
  - `refit_sarima.py` — reloads the full series, refits SARIMA (same order as used in the notebooks), backtests for MAE/RMSE/R², forecasts next month with a 95% CI, writes `dashboard/public/data/forecast.json` and `dashboard/public/data/leaderboard.json`.
  - `export_tda_viz.py` — one-off script (run manually, not in cron) that dumps persistence diagrams, 3D-PCA'd Takens embeddings, and subperiod comparison stats to static JSON (`tda.json`, `embedding3d.json`, `subperiods.json`).
- **`.github/workflows/update-forecast.yml`** — monthly cron (e.g. the 5th) + manual `workflow_dispatch`. Runs fetch → refit → sanity-check → commit → push. Vercel auto-deploys on push to main.

## Data flow (monthly cron cycle)

```
GH Action (monthly cron)
  → fetch_fred.py: pull latest point (needs FRED_API_KEY secret)
      → append to data/input/WPUSI01102B.csv
  → refit_sarima.py: refit SARIMA on full series
      → backtest → MAE / RMSE / R²
      → forecast next month + 95% CI
      → write dashboard/public/data/forecast.json
          (historical actual, fitted, next-month pred + CI, last_updated timestamp)
      → write dashboard/public/data/leaderboard.json
          (SARIMA row refreshed live; RF / LSTM / TDA-feature rows frozen from
          notebook runs, labeled "as of notebook run")
  → sanity check (abort commit if any fail):
      - new row count == old + 1
      - no NaN / inf in forecast or CI
      - CI width > 0
  → git commit + push (only if data changed)
  → Vercel redeploys automatically
```

If the fetch or refit step fails, the workflow fails loudly in the Actions tab and the last-known-good JSON stays live — no silent bad data ever ships.

## Frontend (React + Vite, deployed on Vercel)

Single-page dashboard with section navigation. Visual bar: smooth transitions, gradient-filled CI bands, animated persistence diagram, orbit-controls on the 3D embedding — built following the `dataviz` skill's design system for consistent, polished charts in light/dark.

1. **Forecast** — line chart: historical actual + SARIMA fitted line + next-month projection with CI band (recharts). Stat tiles: latest value, next-month prediction, current model MAE. "Last updated" badge sourced from `forecast.json`'s timestamp; shows a "stale" indicator if older than ~40 days (cron missed).
2. **Model Leaderboard** — sortable table (MAE / RMSE / R² / improvement vs. RF baseline), live SARIMA row visually highlighted and distinguished from frozen rows.
3. **TDA Explorer** — persistence diagram (birth/death scatter, H₀/H₁ colored) and a 3D Takens-embedding scatter (react-three-fiber), colored by subperiod.
4. **Subperiod Comparison** — grouped bars / small multiples showing H₁ persistence strength across 2008–2012 / 2013–2019 / 2020–2026, surfacing the project's key finding (elevated nonlinear cyclicity during the 2008–2012 financial crisis).

`dashboard/src/lib/data.ts` fetches the static JSON files client-side on load; each section component takes a typed JSON shape as props.

## Error handling & testing

- **GH Action:** any failure in fetch or refit aborts before commit; last-known-good JSON keeps serving. Sanity checks (row count, NaN/inf, CI width) enforced in `refit_sarima.py` before it's allowed to write output.
- **Frontend:** stale-data badge if `last_updated` > ~40 days old, instead of failing silently.
- **Testing:** `refit_sarima.py` gets a small pytest sanity test (mock series in, check output shape / no-NaN) run in CI before the commit step is reached. React side: `tsc --noEmit` as a CI gate; no deep test suite needed for a course-project dashboard.

## Secrets / setup required from user

- `FRED_API_KEY` — GitHub Actions secret, needed for `fetch_fred.py`. User must obtain a free key from FRED and add it to repo secrets before the workflow can run.
- Vercel project linked to `dashboard/` subdirectory for deploy.
