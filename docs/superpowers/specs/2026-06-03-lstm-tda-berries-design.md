# LSTM + TDA Forecasting — Berry PPI (Sesión 6)

**Date:** 2026-06-03
**Dataset:** WPUSI01102B.csv — FRED Producer Price Index for Berries, monthly, Jun 2008–Apr 2026 (215 points)

## Goal

Replace the RandomForest forecaster from Session 5 with a stacked multivariate LSTM that ingests TDA features as temporal input channels. Produce a direct metric comparison (MAE, RMSE, R²) between RF baseline and LSTM+TDA on the same held-out test set.

---

## Data Pipeline

### Stage 1: Rolling TDA feature extraction

For each month index `t` in `[35, 214]` (180 steps total):

1. Slice `prices[t-35 : t+1]` — 36 values (W=36, 3 full seasonal cycles)
2. `SingleTakensEmbedding(parameters_type="fixed", dimension=6, time_delay=3)` → embedding shape **(21, 6)**
3. `VietorisRipsPersistence(homology_dimensions=[0, 1])` → persistence diagram
4. Extract features:
   - `PersistenceEntropy` → 2 scalars (H0, H1)
   - `Amplitude` → 2 scalars (H0, H1)
   - Max H1 persistence → 1 scalar (`max(death - birth)` for H1 points)
   - `BettiCurve(n_bins=10)` → 20 scalars (10 per homology dim)
   - **Raw Takens embedding** → `embedding.flatten()` → 126 scalars (21 × 6)
5. Prepend `price_t` → feature vector of length **152** (1+2+2+1+20+126)

Output: matrix `F` of shape `(180, 152)`.

### Stage 2: LSTM sequence construction

Sliding window of length 12 over `F`:
- Input `X[i]` = `F[i : i+12]` — shape `(12, 152)`
- Target `y[i]` = `prices[W + i + L - 1]` = `prices[47 + i]` — next month's price after the window

Total samples: **168**

### Train/val/test split (chronological, no shuffle)

| Split | Indices | Approx samples |
|-------|---------|---------------|
| Train | 0–116   | 117 |
| Val   | 117–141 | 25  |
| Test  | 142–167 | 26  |

**Normalization:** `StandardScaler` fit on train `X` only, applied to val and test. Target `y` normalized separately (same rule).

---

## Model Architecture

```
Input:  (batch, 12, 152)
        ↓
LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
        ↓
LSTM(32, dropout=0.2, recurrent_dropout=0.1)
        ↓
Dense(16, activation='relu')
        ↓
Dense(1)   # predicted price_{t+1} (normalized scale)
```

**Training config:**

| Parameter | Value |
|-----------|-------|
| Loss | **MAE** (avoids mean-collapse; more robust than MSE on small datasets) |
| Optimizer | Adam, lr=0.001 |
| Batch size | 16 |
| Max epochs | 200 |
| Early stopping | patience=20, monitor=val_loss, restore_best_weights=True |

---

## Baseline

RandomForest on same train/test split. Input = flattened `X` (12×152 → 1824 features). Same `y`. Hyperparams: `n_estimators=100` (matches Session 5).

---

## Evaluation

Metrics computed on test set after inverse-transforming predictions:

- MAE
- RMSE
- R²

Displayed as a comparison table:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| RandomForest baseline | | | |
| LSTM + TDA | | | |

Additional plot: actual vs predicted prices on the test period (both models on same axes).

---

## Notebook Structure (`Sesion-6-LSTM-TDA.ipynb`)

1. **Imports & data load** — numpy, pandas, matplotlib, gtda, sklearn, tensorflow/keras
2. **Rolling TDA feature extraction** — store result in `F` array (cache with `np.save`)
3. **Sequence construction & split** — build `X`, `y`; apply StandardScaler
4. **RF baseline** — fit, predict, compute metrics
5. **LSTM model** — define, train, plot training/validation loss curves
6. **Evaluation & comparison** — metrics table + forecast plot
7. **Analysis questions** — markdown reflection prompts (same style as Sessions 2–5)

---

## Implementation Notes

- TDA loop is the bottleneck (~180 VR computations). Cache `F` with `np.save('F_tda_features.npy', F)` after first run; reload with `np.load` to skip recomputation.
- If `F_tda_features.npy` exists from a prior run with N_FEATURES=26, delete it — shape will mismatch (now 152).
- Betti curves: `BettiCurve(n_bins=10)` — 10 bins × 2 dims = 20 scalars.
- Raw embedding appended last: `embedding.flatten()` = 21×6 = 126 scalars per feature vector.
- Target scaler must be fit on `y_train` only; invert-transform both RF and LSTM predictions before computing metrics.
- Use `tensorflow.keras` (not standalone keras) for compatibility.
