# Sesion-8: LSTM+TDA v2 + Exógenas — Design Spec

**Date:** 2026-06-07  
**Notebook:** `Sesion-8-LSTM-TDA-Exog.ipynb`  
**Goal:** Maximize forecast accuracy (minimize MAE) on FRED Berry PPI via ablation study combining improved TDA features, exogenous macro/agricultural variables, and LSTM with self-attention.

---

## 1. Data Pipeline

**Target series:** `WPUSI01102B.csv` — 215 monthly points, Jun 2008 – Apr 2026

**Exogenous series (fredapi + yfinance):**

| Variable | Source | FRED Code / Ticker |
|----------|--------|-------------------|
| S&P500 | yfinance | `^GSPC` (resample month-end → MS) |
| CPI All Items | FRED | `CPIAUCSL` |
| Fed Funds Rate | FRED | `FEDFUNDS` |
| USD Broad Index | FRED | `DTWEXBGS` |
| PPI Fresh Fruits | FRED | `WPU01110301` |
| PPI Vegetables | FRED | `WPS011` |
| Crude Oil WTI | FRED | `DCOILWTICO` |
| Agricultural PPI | FRED | `PPIFAS` |

**Alignment rules:**
- Resample all series to monthly frequency (`MS`)
- Outer join on date index → forward-fill max 2 months for small gaps
- Clip to target series date range
- If any series has >10% missing after ffill → exclude and log warning
- Output: `df_exog` shape `(215, 9)` (target + 8 exogenous)
- `StandardScaler` fit on train split only

---

## 2. TDA Features v2

**Rolling window:** W=36 months, D=6, TAU=3 (same as Sesion-6)  
**VietorisRips:** `homology_dimensions=[0, 1, 2]` (adds H₂)

**Feature vector per timestep (157 total):**

| Block | Features | Count |
|-------|----------|-------|
| PersistenceEntropy H₀, H₁, H₂ | scalar each | 3 |
| Amplitude H₀, H₁, H₂ | scalar each | 3 |
| Max H₁ persistence | scalar | 1 |
| BettiCurve 10-bins × H₀, H₁ | 10×2 | 20 |
| Wasserstein distance diag(t) vs diag(t-1) for H₀, H₁ (set to 0 for t=first window) | 2 scalars | 2 |
| H₁ points with persistence > p75 threshold | scalar | 1 |
| Raw Takens embedding flattened (21 pts × 6 dims) | 126 | 126 |
| **Total** | | **157** |

**Cache:** save to `F_tda_v2_features.npy` (reload if exists to skip recomputation)

---

## 3. LSTM+Attention Architecture

**Input shape:** `(batch, L=12, 165)` — 157 TDA + 8 exogenous per timestep

```
Input (batch, 12, 165)
    ↓
LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    ↓
MultiHeadAttention(num_heads=4, key_dim=16)
    ↓
LayerNormalization
    ↓
LSTM(32, dropout=0.2, recurrent_dropout=0.1)
    ↓
Dense(32, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(16, activation='relu')
    ↓
Dense(1)
```

**Training:**
- Loss: MAE
- Optimizer: Adam(lr=0.001)
- ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5)
- EarlyStopping(patience=30, restore_best_weights=True)
- Split: 70/15/15 chronological on 180 available points (215 − W=35)

---

## 4. Ablation Models

| # | Model | Input | Notes |
|---|-------|-------|-------|
| 1 | RF baseline | TDA v1 (152 feat) | Copy result from Sesion-6 |
| 2 | LSTM puro | prices L=12, shape (batch,12,1) | Train fresh |
| 3 | LSTM+TDA v1 | 152 feat | Copy result from Sesion-6 |
| 4 | LSTM+Exógenas | prices + 8 exog = 9 feat | No TDA |
| 5 | LSTM+TDA v2 | 157 feat, no exog, no attention | Isolate TDA v2 contribution |
| 6 | LSTM+TDA v2+Exog+Attention | 165 feat | **Main model** |
| 7 | LightGBM+TDA v2+Exog | 165 feat flattened (12×165=1980) | `n_estimators=500`, `lr=0.05` |

---

## 5. Analysis Outputs

**§7 Comparison table:** MAE, RMSE, R² for all 7 models on same test set + bar chart

**§8 SHAP analysis** (on LightGBM model 7):
- Top-20 feature importances
- Identify which exogenous series and TDA features drive predictions most

**§9 Discussion questions:**
1. Does TDA v2 improve over TDA v1?
2. Do exogenous variables add value beyond TDA alone?
3. Which exogenous series has highest SHAP importance?
4. Does attention mechanism help given N=180?

---

## 6. Notebook Structure

```
§0  Imports (fredapi, yfinance, lightgbm, shap, tensorflow, gtda)
§1  Download & align exogenous series
§2  TDA features v2 — rolling extraction + cache
§3  Sequence construction (L=12) + train/val/test split
§4  Ablation models 2, 4, 5 — LSTM variants
§5  Main model 6 — LSTM+TDA v2+Exog+Attention
§6  Model 7 — LightGBM
§7  Comparison table + forecast plot
§8  SHAP feature importance
§9  Analysis questions
```

---

## 7. Dependencies

```bash
pip install fredapi yfinance lightgbm shap giotto-tda tensorflow scikit-learn plotly
```

FRED API key: hardcode in §0 cell as `FRED_KEY = "your_key_here"` (user has key — do not commit key to git).
