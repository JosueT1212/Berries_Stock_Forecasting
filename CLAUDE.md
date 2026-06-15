# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

University course project (6th semester, Topology) analyzing the FRED Producer Price Index for Berries (`WPUSI01102B`) using Topological Data Analysis (TDA). The analysis applies Takens embedding and persistent homology to detect seasonality and structural breaks in the monthly price series (June 2008 – April 2026).

## Running Notebooks

```bash
jupyter notebook
# or
jupyter lab
```

Run individual cells sequentially — each notebook session builds state (embeddings, diagrams) that later cells depend on.

## Key Libraries

| Library | Purpose |
|---------|---------|
| `giotto-tda` (`gtda`) | Takens embedding, Vietoris-Rips persistence, diagram featurization |
| `ripser` / `persim` | Alternative persistence computation and diagram plotting |
| `sklearn` | PCA, StandardScaler, RandomForestRegressor, metrics |
| `tensorflow` / `keras` | LSTM forecasting model (Sesion-6) |
| `plotly` | Interactive visualizations in later sessions |

Install: `pip install giotto-tda ripser persim scikit-learn plotly tensorflow`

## Folder Structure

```
notebooks/          — all Sesion-*.ipynb analysis notebooks
data/
  input/            — WPUSI01102B.csv (raw FRED data)
  output/           — .npy feature caches + output CSVs
presentations/
  images/           — Tec_Logo.jpeg and slide assets
  *.pdf/.pptx/.py   — slides and generator scripts
report/
  images/           — figures for the LaTeX report
  reporte_tda_berries.tex
```

Notebooks use relative paths (`../data/input/`, `../data/output/`). Presentation scripts use the same convention from their directory.

## Data

`data/input/WPUSI01102B.csv` — two columns: `observation_date` (monthly), `WPUSI01102B` (PPI index, base 1982=100). Load with `parse_dates=['observation_date']`.

Cached features: `data/output/F_tda_features.npy` (152 features), `data/output/F_tda_v2_features.npy` (157 features).

## Notebook Progression

- **Sesion-2-R2-1.ipynb** — EDA, subperiod definition (2008-2012, 2013-2019, 2020-2026), Takens embedding, persistence diagrams
- **Sesion-3-R2.ipynb** — Topological properties of the data
- **Sesion-4-R2-E.ipynb** — Parameter optimization for TDA (dimension `d`, time delay `τ`)
- **Sesion-5-R2-E.ipynb** — Full period optimization + TDA-enhanced forecasting with RandomForest
- **Sesion-6-LSTM-TDA.ipynb** — LSTM forecasting with raw Takens embedding + TDA summary features (N_FEATURES=152), MAE loss, RF baseline comparison
- **Sesion-6-R2-E.ipynb** — Extended RF forecasting with calendar lags
- **Sesion-7-LSTM-Pure.ipynb** — Pure LSTM baseline (no TDA features)
- **Sesion-8-LSTM-TDA-Exog.ipynb** — LSTM with TDA v2 exogenous features (157)
- **Sesion-9-EP-Loss-MultiTask.ipynb** — Multi-task learning with EP loss
- **Sesion-10-Calendar-Lag.ipynb** — Calendar and lag feature engineering

## Core TDA Pattern

```python
# Standard pipeline used throughout
STE = SingleTakensEmbedding(parameters_type="fixed", dimension=12, time_delay=3)
embedding = STE.fit_transform(prices)                          # shape (n_points, d)
diagrams = VietorisRipsPersistence(homology_dimensions=[0, 1]).fit_transform([embedding])[0]
```

Key parameters: `dimension=12` (one year of monthly data), `time_delay=3` (quarterly patterns). Normalize with `StandardScaler` before embedding when comparing subperiods.

## Subperiod Analysis

Three subperiods are defined by boolean masks on the numpy array of dates:
- 2008-2012: financial crisis, high volatility, strongest H₁ persistence
- 2013-2019: relative stability
- 2020-2026: pandemic/post-pandemic inflation
