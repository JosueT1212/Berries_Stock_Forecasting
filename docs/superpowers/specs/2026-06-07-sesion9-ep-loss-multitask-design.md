# Sesion-9: EP Loss + Multi-Task Peak Head — Design Spec

**Date:** 2026-06-07  
**Notebook:** `Sesion-9-EP-Loss-MultiTask.ipynb`  
**Goal:** Improve peak/valley prediction on FRED Berry PPI by replacing MAE loss with Enhanced Peak (EP) Loss and adding a multi-task classification head. Beat Sesion-8 best model MAE=27.92 (LSTM+TDA v2).

---

## 1. Data Pipeline — Peak Labels

Peak labels derived from full price series before split (no leakage — supervisory signal):

```python
from scipy.signal import argrelextrema

extrema_idx = set(
    argrelextrema(prices, np.greater, order=3)[0].tolist() +
    argrelextrema(prices, np.less,    order=3)[0].tolist()
)

# y_peak[i] = 1 if prices[W-1 + i + L] is a local extremum
y_peak = np.array([
    1 if (W - 1 + i + L) in extrema_idx else 0
    for i in range(len(F_tda) - L)
])  # shape (168,) — ~19% positives (32 of 168)
```

**Split**: same chronological 70/15/15 as Sesion-8 (train=117, val=25, test=26).  
`y_peak_train=y_peak[:117]`, `y_peak_val=y_peak[117:142]`, `y_peak_test=y_peak[142:]`.  
**F source**: reload `F_tda_v2_features.npy` (cached from Sesion-8, shape (180,157)).  
**Features**: F[:, :157] only — TDA v2 features, no exogenous (matches Sesion-8 best model).

---

## 2. EP Loss + Combined Loss

**Enhanced Peak MAE:**
```python
# weight = 1 + α·is_peak → peaks penalized α+1× more than normal points
ep_mae = mean[(1 + α·is_peak) · |price_true - price_pred|]
```

**Combined loss** (y_true shape: (batch,2), y_pred shape: (batch,2)):
```python
def combined_loss(alpha, beta):
    def loss(y_true, y_pred):
        price_true = y_true[:, 0]
        is_peak    = y_true[:, 1]
        price_pred = y_pred[:, 0]
        peak_pred  = y_pred[:, 1]
        weight  = 1.0 + alpha * is_peak
        ep_mae  = tf.reduce_mean(weight * tf.abs(price_true - price_pred))
        bce     = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        is_peak[:, tf.newaxis], peak_pred[:, tf.newaxis]))
        return ep_mae + beta * bce
    return loss
```

---

## 3. Model Architecture

Base: LSTM+TDA v2 from Sesion-8 (best model, no exogenous, no attention).  
Change: split final Dense into two output heads.

```
Input (batch, 12, 157)
  → LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
  → LSTM(32, dropout=0.2, recurrent_dropout=0.1)
  → Dense(32, relu)
  → Dropout(0.3)
  → Dense(16, relu)  [shared trunk]
  ├→ Dense(1, linear,  name='price_out')    ← price regression
  └→ Dense(1, sigmoid, name='peak_out')     ← peak/valley classification
```

**Compiled with:** `combined_loss(alpha, beta)`, `optimizer=Adam(1e-3)`  
**Callbacks:** `EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)`  
**Epochs:** 300, batch_size=16

---

## 4. Grid Search

**Search space:** α ∈ {2, 3, 5, 8} × β ∈ {0.1, 0.3, 0.5} = **12 combinations**

**For each (α, β):**
1. Build fresh model
2. Train on `(X_train_s, Y_train)` where `Y_train = np.stack([y_train_s, y_peak_train], axis=1)`
3. Evaluate val MAE on price only: `mean_absolute_error(y_val, price_pred_val_descaled)`
4. Store: `{alpha, beta, val_mae}`

**Winner:** combination with minimum val MAE on price (not combined loss).

**Final model:** retrain winner on train+val. Refit `scaler_X` and `scaler_y` on train+val combined before final training. Evaluate on test with these scalers.

---

## 5. Evaluation

**Primary metric:** MAE global on test set — compare to Sesion-8 best (27.92).

**Outputs:**
- §5: Grid search results table (12 rows: α, β, val_mae) + winner highlighted
- §5: Final test MAE vs Sesion-8 comparison
- §6: Forecast plot with peak/valley timestamps marked (vertical lines or scatter)
- §7: Analysis questions

---

## 6. Notebook Structure

```
§0  Imports + constants (reuse W=36, D=6, TAU=3, L=12)
§1  Load F_tda_v2_features.npy + prices + peak labels
§2  Sequence construction + split + normalization (same as Sesion-8)
§3  EP Loss + combined_loss function definition
§4  Build model function (returns fresh Functional API model)
§5  Grid search 12 combos → results table → winner
§6  Final model: retrain winner on train+val → test MAE
§7  Forecast plot with peaks/valleys marked
§8  Analysis questions
```

---

## 7. Dependencies

All available in `/usr/local/bin/python3` (Python 3.11):
- `scipy` (argrelextrema)
- `tensorflow`, `gtda`, `sklearn`, `numpy`, `pandas`, `matplotlib`

**Reuses from Sesion-8 (no recomputation):**
- `F_tda_v2_features.npy` — TDA cache
- Same train/val/test split indices
