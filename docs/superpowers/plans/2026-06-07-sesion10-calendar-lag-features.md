# Sesion-10 Calendar + Lag Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Sesion-10-Calendar-Lag.ipynb` — add month sin/cos and lag features to LSTM+TDA v1 (Sesion-6) to test whether explicit seasonal encoding and lag signals improve on the shift-prone baseline MAE.

**Architecture:** Load cached TDA v1 features (`F_tda_features.npy`, shape 180×152). Build feature-enriched sequences: Variant A appends month_sin/cos (→154 features), Variant B also appends lag-1 + lag-12 prices (→156 features). Same Sequential LSTM(64)→LSTM(32)→Dense(16,relu)→Dense(1) architecture as Sesion-6. Three models evaluated on identical 70/15/15 split.

**Tech Stack:** Python 3.11 (`/usr/local/bin/python3`), TensorFlow 2.21, scikit-learn 1.3.2, numpy, pandas, matplotlib.

---

## Files

- **Create:** `Sesion-10-Calendar-Lag.ipynb` in `/Users/josuetapiahernandez/Documents/6_Semestre/Topología/Berries/`
- **Reads:** `F_tda_features.npy` (TDA v1 cache), `WPUSI01102B.csv`

---

## Key Constants & Formulas (reference for all tasks)

```python
W, D, TAU, L = 36, 6, 3, 12
N_BASE = 152   # TDA v1 features
# F shape: (180, 152)
# Sequence i: X[i] = F[i:i+12],  y[i] = prices[target_idx]
# target_idx = W + i + L - 1 = 47 + i   (i in 0..167)
# month of target = pd.Timestamp(dates[47+i]).month
# lag_1  = prices[46+i]   # one month before target
# lag_12 = prices[35+i]   # 12 months before target
```

---

## Task 1: §0 Imports + §1 Load data + TDA v1 cache

**Files:**
- Create: `Sesion-10-Calendar-Lag.ipynb`

- [ ] **Step 1: Create notebook skeleton with nbformat**

Run this Python script to bootstrap the file:
```python
import nbformat, json

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"}
}
nb.cells = []
with open('/Users/josuetapiahernandez/Documents/6_Semestre/Topología/Berries/Sesion-10-Calendar-Lag.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("Created empty notebook")
```

- [ ] **Step 2: Add title markdown cell (Cell 0)**

```markdown
# Sesión 10: Calendar + Lag Features sobre LSTM+TDA v1

## Objetivo
Mejorar el LSTM+TDA v1 (Sesión 6) agregando:
- **Variant A**: features de mes (sin/cos) — codifica estacionalidad anual explícita
- **Variant B**: Variant A + lag-1 + lag-12 de precio — codifica contexto de nivel

Comparar 3 modelos sobre el mismo split 70/15/15 para aislar el efecto de cada feature group.
```

- [ ] **Step 3: Add §0 imports cell (Cell 1)**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

W, D, TAU, L = 36, 6, 3, 12
N_BASE = 152   # TDA v1 features per timestep

print(f"TF {tf.__version__}")
```

Expected: `TF 2.21.0`

- [ ] **Step 4: Add §1 markdown cell (Cell 2)**

```markdown
## §1 — Carga de Datos y Features TDA v1
```

- [ ] **Step 5: Add §1 data + cache load cell (Cell 3)**

```python
df     = pd.read_csv('WPUSI01102B.csv', parse_dates=['observation_date'])
prices = df['WPUSI01102B'].values.astype(float)
dates  = df['observation_date'].values
print(f"Prices: {len(prices)} pts  [{prices.min():.1f}, {prices.max():.1f}]")

# Load TDA v1 cache (saved by Sesion-6)
F = np.load('F_tda_features.npy')   # shape (180, 152)
assert F.shape == (len(prices) - W + 1, N_BASE), f"Unexpected shape: {F.shape}"
print(f"F_tda v1: {F.shape}  (price[t] is F[:,0])")
# F[t,0] = prices[W-1+t] = last price in window t
# Sequence i: X[i] = F[i:i+12], target = prices[47+i]
n_seq = len(F) - L   # = 168
print(f"Sequences: {n_seq}")
assert n_seq == 168
```

Expected:
```
Prices: 215 pts  [...]
F_tda v1: (180, 152)
Sequences: 168
```

---

## Task 2: §2 Feature engineering + §3 Sequences + split + normalize

- [ ] **Step 1: Add §2 markdown cell (Cell 4)**

```markdown
## §2 — Feature Engineering: Calendar + Lag

**Variant A** appends `month_sin`, `month_cos` of the TARGET month → 154 features.  
**Variant B** also appends `lag_1` (price 1 month before target) and `lag_12` (same month last year) → 156 features.  
All extra features broadcast constant across the L=12 timesteps of each sequence.
```

- [ ] **Step 2: Add §2 feature engineering cell (Cell 5)**

```python
X_base_list, X_A_list, X_B_list, y_list = [], [], [], []

for i in range(n_seq):
    target_idx = 47 + i   # = W + i + L - 1

    # ── Calendar features (target month) ─────────────────
    month = pd.Timestamp(dates[target_idx]).month
    m_sin = np.sin(2 * np.pi * month / 12)
    m_cos = np.cos(2 * np.pi * month / 12)

    # ── Lag features (raw price, same scale as y) ─────────
    lag_1  = prices[target_idx - 1]    # = prices[46+i]
    lag_12 = prices[target_idx - 12]   # = prices[35+i]

    # ── Base sequence ─────────────────────────────────────
    base_seq = F[i : i + L]            # (12, 152)

    # ── Extra features broadcast to all L timesteps ───────
    cal_broadcast  = np.tile([m_sin, m_cos],             (L, 1))  # (12, 2)
    lag_broadcast  = np.tile([m_sin, m_cos, lag_1, lag_12], (L, 1))  # (12, 4)

    X_base_list.append(base_seq)                                          # (12, 152)
    X_A_list.append(np.concatenate([base_seq, cal_broadcast], axis=1))   # (12, 154)
    X_B_list.append(np.concatenate([base_seq, lag_broadcast], axis=1))   # (12, 156)
    y_list.append(prices[target_idx])

X_base = np.array(X_base_list)   # (168, 12, 152)
X_A    = np.array(X_A_list)      # (168, 12, 154)
X_B    = np.array(X_B_list)      # (168, 12, 156)
y      = np.array(y_list)        # (168,)

assert X_base.shape == (168, 12, 152)
assert X_A.shape    == (168, 12, 154)
assert X_B.shape    == (168, 12, 156)
print(f"X_base {X_base.shape}  X_A {X_A.shape}  X_B {X_B.shape}")
print(f"y range: [{y.min():.1f}, {y.max():.1f}]")
```

Expected:
```
X_base (168, 12, 152)  X_A (168, 12, 154)  X_B (168, 12, 156)
y range: [...]
```

- [ ] **Step 3: Add §3 markdown cell (Cell 6)**

```markdown
## §3 — Split + Normalización

Split cronológico 70/15/15. Scalers fit solo en train.
```

- [ ] **Step 4: Add §3 split + normalize cell (Cell 7)**

```python
n_train = int(n_seq * 0.70)   # 117
n_val   = int(n_seq * 0.15)   # 25
n_test  = n_seq - n_train - n_val  # 26

def split_scale(X, n_train, n_val, n_feat):
    X_tr, X_v, X_te = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_v_s  = sc.transform(X_v.reshape(-1, n_feat)).reshape(X_v.shape)
    X_te_s = sc.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape)
    return X_tr_s, X_v_s, X_te_s

X_base_tr, X_base_v, X_base_te = split_scale(X_base, n_train, n_val, 152)
X_A_tr,    X_A_v,    X_A_te    = split_scale(X_A,    n_train, n_val, 154)
X_B_tr,    X_B_v,    X_B_te    = split_scale(X_B,    n_train, n_val, 156)

y_train = y[:n_train]
y_val   = y[n_train:n_train+n_val]
y_test  = y[n_train+n_val:]

scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_s   = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

# Test dates for plot
test_start_t = 47 + n_train + n_val
test_dates   = dates[test_start_t : test_start_t + n_test]

print(f"Train={n_train}  Val={n_val}  Test={n_test}")
print("Normalización ✓")
```

Expected: `Train=117  Val=25  Test=26`

---

## Task 3: §4 Baseline retrain + §5 Model A

- [ ] **Step 1: Add build_lstm helper + §4 markdown cell (Cell 8)**

```markdown
## §4 — Baseline: LSTM+TDA v1 Reentrenado (152 features)

Mismo modelo que Sesión 6 — reentrenado en el mismo split para comparación honesta.
```

- [ ] **Step 2: Add §4 build helper + baseline cell (Cell 9)**

```python
def build_lstm(n_features, name='LSTM_TDA'):
    model = Sequential([
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1,
             input_shape=(L, n_features)),
        LSTM(32, dropout=0.2, recurrent_dropout=0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ], name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mae')
    return model

def train_eval(X_tr, X_v, X_te, name, n_features):
    tf.random.set_seed(42); np.random.seed(42)
    model = build_lstm(n_features, name)
    cb = EarlyStopping(monitor='val_loss', patience=20,
                       restore_best_weights=True, verbose=0)
    model.fit(X_tr, y_train_s, validation_data=(X_v, y_val_s),
              epochs=200, batch_size=16, callbacks=[cb], verbose=0)
    preds_s = model.predict(X_te, verbose=0).flatten()
    preds   = scaler_y.inverse_transform(preds_s.reshape(-1,1)).flatten()
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"{name:30s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return preds, mae, rmse, r2

preds_base, mae_base, rmse_base, r2_base = train_eval(
    X_base_tr, X_base_v, X_base_te, 'Baseline (152)', 152)
```

Expected: one result line `Baseline (152)  MAE=XX.XXXX  RMSE=XX.XXXX  R²=X.XXXX`

- [ ] **Step 3: Add §5 markdown cell (Cell 10)**

```markdown
## §5 — Model A: LSTM+TDA v1 + Calendar (154 features)

Agrega `month_sin` y `month_cos` del mes objetivo. El modelo ahora "sabe" en qué mes del año se hace la predicción.
```

- [ ] **Step 4: Add §5 Model A cell (Cell 11)**

```python
preds_A, mae_A, rmse_A, r2_A = train_eval(
    X_A_tr, X_A_v, X_A_te, 'Model A +Calendar (154)', 154)
```

Expected: `Model A +Calendar (154)  MAE=XX.XXXX  RMSE=XX.XXXX  R²=X.XXXX`

---

## Task 4: §6 Model B + §7 Comparison + §8 Analysis

- [ ] **Step 1: Add §6 markdown cell (Cell 12)**

```markdown
## §6 — Model B: LSTM+TDA v1 + Calendar + Lag (156 features)

Agrega además `lag_1` (precio un mes antes del objetivo) y `lag_12` (mismo mes año anterior).  
Nota: `lag_1` equivale a `F[i+11,0]` que ya está en las 152 features base, pero explícito  
en canal separado + `lag_12` añade información de estacionalidad interanual.
```

- [ ] **Step 2: Add §6 Model B cell (Cell 13)**

```python
preds_B, mae_B, rmse_B, r2_B = train_eval(
    X_B_tr, X_B_v, X_B_te, 'Model B +Cal+Lag (156)', 156)
```

Expected: `Model B +Cal+Lag (156)  MAE=XX.XXXX  RMSE=XX.XXXX  R²=X.XXXX`

- [ ] **Step 3: Add §7 markdown cell (Cell 14)**

```markdown
## §7 — Comparación + Forecast Plot
```

- [ ] **Step 4: Add §7 comparison table + plot cell (Cell 15)**

```python
# ── Comparison table ──────────────────────────────────────
df_res = pd.DataFrame({
    'Modelo':   ['Baseline (152)', 'Model A +Calendar (154)', 'Model B +Cal+Lag (156)'],
    'Features': [152, 154, 156],
    'MAE':      [mae_base, mae_A, mae_B],
    'RMSE':     [rmse_base, rmse_A, rmse_B],
    'R²':       [r2_base, r2_A, r2_B],
    'Δ MAE':    [0.0, mae_A - mae_base, mae_B - mae_base]
})
print("\n" + "="*65)
print("  COMPARACIÓN FINAL — Test Set")
print("="*65)
print(df_res.to_string(index=False, float_format='{:.4f}'.format))
print("="*65)

# ── Forecast plot ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(test_dates, y_test,    'k-',  linewidth=2,   label='Real', zorder=5)
ax.plot(test_dates, preds_base,'b--', linewidth=1.5, label=f'Baseline  MAE={mae_base:.2f}')
ax.plot(test_dates, preds_A,   'g--', linewidth=1.5, label=f'+Calendar MAE={mae_A:.2f}')
ax.plot(test_dates, preds_B,   'r--', linewidth=1.5, label=f'+Cal+Lag  MAE={mae_B:.2f}')
ax.set_xlabel('Fecha')
ax.set_ylabel('PPI Berries (1982=100)')
ax.set_title('Sesión 10 — Calendar + Lag Features\nComparación en Test Set')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Add §8 analysis questions markdown cell (Cell 16)**

```markdown
## §8 — Preguntas de Análisis

1. **¿Calendar features mejoraron MAE?** Compara Baseline vs Model A (Δ MAE). ¿El LSTM aprovecha la señal estacional del mes objetivo, o ya la infería del embedding TDA?

2. **¿Lag-1 + lag-12 añadieron valor sobre el calendario?** Compara Model A vs Model B. Nota: lag-1 ya está implícito en F[:,0] del último timestep. ¿La redundancia explícita ayudó o fue ruido?

3. **¿Visualmente las predicciones siguen "shifteadas"?** Observa el forecast plot. ¿Model B, que tiene lag-1 explícito, muestra un shift más pronunciado o menos que el Baseline?

4. **¿Cuál feature group aporta más?** TDA topológico (152), calendario (2) o lags (2). ¿Qué esperarías con N=168 muestras?

5. **Comparación histórica:** Sesión 8 LSTM+TDA v2 alcanzó MAE=27.92. ¿Los features calendario/lag en v1 alcanzan ese nivel? ¿Qué explica la diferencia?
```

- [ ] **Step 6: Execute notebook end-to-end**

Run:
```bash
cd /Users/josuetapiahernandez/Documents/6_Semestre/Topología/Berries && \
/usr/local/bin/python3 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  --ExecutePreprocessor.kernel_name=python3 \
  Sesion-10-Calendar-Lag.ipynb 2>&1
```

Expected: `[NbConvertApp] Writing XXXXXX bytes to Sesion-10-Calendar-Lag.ipynb`

- [ ] **Step 7: Verify results printed**

```bash
cd /Users/josuetapiahernandez/Documents/6_Semestre/Topología/Berries && \
/usr/local/bin/python3 -c "
import json
with open('Sesion-10-Calendar-Lag.ipynb') as f:
    nb = json.load(f)
for c in nb['cells']:
    for o in c.get('outputs', []):
        t = o.get('text', [])
        if isinstance(t, list): t = ''.join(t)
        if 'COMPARACIÓN FINAL' in t:
            print(t)
"
```

Expected: prints the comparison table with MAE for all 3 models.

---

## Self-Review

**Spec coverage:**
- ✓ §1 Load F_tda_features.npy (180,152)
- ✓ §2 month_sin/cos for target month, lag_1=prices[46+i], lag_12=prices[35+i], broadcast to L=12 timesteps
- ✓ §3 X_base(168,12,152), X_A(168,12,154), X_B(168,12,156), split 117/25/26, scaler fit on train
- ✓ §4 Baseline: build_lstm(152), same architecture as Sesion-6 (LSTM64→LSTM32→Dense16→Dense1)
- ✓ §5 Model A: build_lstm(154)
- ✓ §6 Model B: build_lstm(156)
- ✓ §7 Comparison table with Δ MAE column + 3-line forecast plot
- ✓ §8 5 analysis questions

**Name consistency:** `build_lstm`, `train_eval`, `split_scale`, `X_base/X_A/X_B`, `preds_base/preds_A/preds_B`, `mae_base/mae_A/mae_B` — consistent across all tasks.

**Architecture match:** Dense(16, relu) before Dense(1) — matches actual Sesion-6 code (not just LSTM→Dense(1)).

**No placeholders:** All code complete, all expected outputs specified.
