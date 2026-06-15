# Sesion-10: Calendar + Lag Features on LSTM+TDA v1 — Design Spec

**Date:** 2026-06-07  
**Notebook:** `Sesion-10-Calendar-Lag.ipynb`  
**Goal:** Improve Berry PPI LSTM+TDA v1 forecasting (Sesion-6 baseline) by adding explicit calendar features (month sin/cos) and lag features (lag-1, lag-12) to fix the implicit shift bias and encode annual seasonality. Compare two enriched variants vs baseline in one notebook.

---

## 1. Feature Engineering

Base: TDA v1 features (152 per timestep) recomputed from scratch using same parameters as Sesion-6:
- `SingleTakensEmbedding(d=6, τ=3)` → raw Takens 21×6 = 126 features
- `VietorisRipsPersistence(homology_dims=[0,1])` → PE(2) + AMP(2) + maxH1(1) + Betti(4) + misc = 26 features
- Total: 152 features per window position

**Variant A — Calendar only (154 features):**
```python
month_target = pd.Timestamp(dates[W - 1 + i + L]).month  # month of the predicted price
month_sin = np.sin(2 * np.pi * month_target / 12)
month_cos = np.cos(2 * np.pi * month_target / 12)
# Appended as constant across all L=12 timesteps in the sequence
# X_A[i] shape: (12, 154)
```

**Variant B — Calendar + Lags (156 features):**
```python
lag_1  = prices[W - 1 + i + L - 1]   # price one month before target (normalized)
lag_12 = prices[W - 1 + i + L - 12]  # price 12 months before target (normalized)
# X_B[i] shape: (12, 156)
```

Both month_sin/cos and lag features are constant across the L=12 timesteps of each sequence (broadcast). Lag features are normalized with the same `scaler_y` used for targets.

---

## 2. Architecture + Training

Same LSTM architecture as Sesion-6, only `n_features` changes:

```
Baseline : Input(12, 152) → LSTM(64, rs=True) → LSTM(32) → Dense(1)
Model A  : Input(12, 154) → LSTM(64, rs=True) → LSTM(32) → Dense(1)
Model B  : Input(12, 156) → LSTM(64, rs=True) → LSTM(32) → Dense(1)
```

Training config (all models identical):
- Loss: MAE
- Optimizer: Adam(lr=1e-3)
- Callbacks: EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True) + ReduceLROnPlateau(patience=15, factor=0.5)
- Epochs: 400, batch_size=16
- Split: 70/15/15 chronological → train=117, val=25, test=26
- Scalers: StandardScaler fit on train only for both X and y

---

## 3. Notebook Structure

```
§0  Imports + constants (W=36, D=6, TAU=3, L=12, N_BASE=152)
§1  Load prices + recompute TDA v1 features (152) — sliding window W=36
§2  Feature engineering: month sin/cos + lag-1 + lag-12 per sequence
§3  Sequence construction + split + normalization
    - X_base (168,12,152), X_A (168,12,154), X_B (168,12,156)
    - y (168,), y_peak not needed
§4  Baseline retrain: LSTM+TDA v1 (152 feat) on same split
§5  Model A: LSTM+TDA v1 + Calendar (154 feat) → test MAE
§6  Model B: LSTM+TDA v1 + Calendar + Lag (156 feat) → test MAE
§7  Comparison table + forecast plot (3 models + actual)
§8  Analysis questions
```

---

## 4. Evaluation

**Primary metric:** Test MAE (original price scale, descaled with `scaler_y`).

**Comparison table:**

| Modelo | Features | Test MAE | Δ vs Baseline |
|--------|----------|----------|---------------|
| Baseline (TDA v1) | 152 | — | — |
| +Calendar (A) | 154 | — | ±X.XX |
| +Calendar+Lag (B) | 156 | — | ±X.XX |

**Forecast plot:** 3 predicted lines + actual, test period only.

**Analysis questions (§8):**
1. ¿Calendar features mejoraron MAE? ¿El modelo aprovecha la estacionalidad anual?
2. ¿Lag-1 + lag-12 explícitos mejoran sobre solo calendario? ¿O el LSTM ya los infería implícitamente?
3. ¿Las predicciones de Model B siguen "shifteadas" visualmente, o el lag-1 explícito cambia el patrón?
4. ¿Qué feature group (TDA vs calendario vs lag) crees que aporta más según los resultados?

---

## 5. Dependencies

All in `/usr/local/bin/python3` (Python 3.11):
- `gtda` (SingleTakensEmbedding, VietorisRipsPersistence, PersistenceEntropy, Amplitude)
- `tensorflow` 2.21, `sklearn` 1.3.2, `numpy`, `pandas`, `matplotlib`

**No cache reuse** — TDA v1 features recomputed fresh (no `F_tda_v2_features.npy` used).
