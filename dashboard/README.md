# Berries PPI Forecast Dashboard

A React + TypeScript dashboard that visualizes the SARIMA forecast for the
Berries Producer Price Index (PPI), along with its topological-data-analysis
(TDA) diagnostics and model leaderboard.

## Run locally

```bash
npm install
npm run dev
```

## Data

The dashboard reads static JSON from `public/data/*.json`:

- `forecast.json` — SARIMA forecast, history, and confidence intervals, written by `../scripts/refit_sarima.py`
- `leaderboard.json` — model comparison metrics, written by `../scripts/refit_sarima.py`
- TDA visualization data, written by `../scripts/export_tda_viz.py`

These files are refreshed monthly by `.github/workflows/update-forecast.yml`,
which fetches the latest FRED observation, refits the model, and commits the
updated JSON.

See the root [README](../README.md#live-dashboard) for full project setup,
deployment, and forecasting pipeline details.
