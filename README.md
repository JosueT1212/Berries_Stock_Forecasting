# TDA Analysis of Berry Producer Price Index

**Course:** Topology — 6th Semester  
**Dataset:** FRED `WPUSI01102B` — Producer Price Index for Berries (monthly, June 2008 – April 2026, base 1982=100)

## Overview

Applies Topological Data Analysis (TDA) to a commodity price time series. Pipeline: Takens delay embedding → Vietoris-Rips persistent homology → topological feature extraction → ML/DL forecasting. Goal: detect seasonality and structural breaks invisible to classical methods.

## Methods

- **Takens embedding** — reconstructs attractor from scalar time series (`d=12`, `τ=3`)
- **Persistent homology** — H₀ (connected components) and H₁ (loops) via Vietoris-Rips
- **Subperiod comparison** — 2008–2012 (financial crisis), 2013–2019 (stability), 2020–2026 (pandemic/inflation)
- **Forecasting** — Random Forest with TDA features, LSTM with raw embeddings, multi-task learning with EP loss

## Structure

```
notebooks/          — analysis sessions (run sequentially)
data/
  input/            — WPUSI01102B.csv (raw FRED data)
  output/           — cached TDA features (.npy), model predictions (.csv)
presentations/      — slides and generator scripts
report/
  reporte_tda_berries.tex
```

## Notebooks

| Notebook | Content |
|---|---|
| `Sesion-2-R2-1.ipynb` | EDA, subperiod definition, Takens embedding, persistence diagrams |
| `Sesion-3-R2.ipynb` | Topological properties analysis |
| `Sesion-4-R2-E.ipynb` | Parameter optimization (`d`, `τ`) |
| `Sesion-5-R2-E.ipynb` | Full-period TDA + RandomForest forecasting |
| `Sesion-6-LSTM-TDA.ipynb` | LSTM with 152 TDA summary features, MAE loss |
| `Sesion-6-R2-E.ipynb` | RF forecasting with calendar lags |
| `Sesion-7-LSTM-Pure.ipynb` | Pure LSTM baseline (no TDA) |
| `Sesion-8-LSTM-TDA-Exog.ipynb` | LSTM with 157 TDA v2 exogenous features |
| `Sesion-9-EP-Loss-MultiTask.ipynb` | Multi-task learning with EP loss |
| `Sesion-10-Calendar-Lag.ipynb` | Calendar and lag feature engineering |

## Setup

```bash
pip install giotto-tda ripser persim scikit-learn plotly tensorflow
jupyter lab
```

Run each notebook top-to-bottom — cells build shared state (embeddings, diagrams) used by later cells.

## Core Pipeline

```python
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence

STE = SingleTakensEmbedding(parameters_type="fixed", dimension=12, time_delay=3)
embedding = STE.fit_transform(prices)          # (n_points, 12)

diagrams = VietorisRipsPersistence(
    homology_dimensions=[0, 1]
).fit_transform([embedding])[0]                # birth/death pairs for H₀ and H₁
```

## Key Finding

H₁ persistence (loops in the attractor) is strongest during 2008–2012, confirming elevated nonlinear cyclicity during the financial crisis. TDA features improve forecasting MAE over pure-LSTM baselines.

## Data

`data/input/WPUSI01102B.csv` — two columns: `observation_date` (monthly), `WPUSI01102B` (PPI value).

```python
import pandas as pd
df = pd.read_csv('data/input/WPUSI01102B.csv', parse_dates=['observation_date'])
```

Cached features: `F_tda_features.npy` (152 features), `F_tda_v2_features.npy` (157 features).
