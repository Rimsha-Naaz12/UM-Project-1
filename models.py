"""
STEP 2 - Forecasting Models: Baseline + Statistical + ML
UAC Predictive Forecasting Project
Run: python step2_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("=" * 60)
print("STEP 2: Forecasting Models")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/uac_cleaned.csv", index_col='Date', parse_dates=True)
target = df['hhs_care'].dropna()

# ─── TRAIN/TEST SPLIT (last 90 days = test) ───────────────────────────────────
HORIZON = 30   # forecast horizon for future predictions
TEST_DAYS = 90

train = target.iloc[:-TEST_DAYS]
test  = target.iloc[-TEST_DAYS:]
print(f"Train: {len(train)} days | Test: {len(test)} days")

def metrics(actual, predicted, name):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"  {name:30s}  MAE={mae:7.1f}  RMSE={rmse:7.1f}  MAPE={mape:5.2f}%")
    return {'model': name, 'MAE': round(mae,1), 'RMSE': round(rmse,1), 'MAPE': round(mape,2)}

results = []

# ─── 1. NAIVE (persist last value) ───────────────────────────────────────────
naive_pred = pd.Series([train.iloc[-1]] * TEST_DAYS, index=test.index)
results.append(metrics(test, naive_pred, "Naive Persistence"))

# ─── 2. MOVING AVERAGE ────────────────────────────────────────────────────────
ma_pred = pd.Series([train.rolling(7).mean().iloc[-1]] * TEST_DAYS, index=test.index)
results.append(metrics(test, ma_pred, "7-Day Moving Average"))

# ─── 3. EXPONENTIAL SMOOTHING ────────────────────────────────────────────────
from statsmodels.tsa.holtwinters import ExponentialSmoothing
es_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()
es_pred = es_model.forecast(TEST_DAYS)
es_pred.index = test.index
results.append(metrics(test, es_pred, "Exponential Smoothing"))

# ─── 4. ARIMA ────────────────────────────────────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(train, order=(5, 1, 2)).fit()
arima_pred = arima_model.forecast(TEST_DAYS)
arima_pred.index = test.index
results.append(metrics(test, arima_pred, "ARIMA(5,1,2)"))

# ─── 5. ML MODELS ────────────────────────────────────────────────────────────
feature_cols = [
    'cbp_apprehended', 'cbp_transfers', 'hhs_discharged', 'net_pressure',
    'hhs_care_lag1', 'hhs_care_lag7', 'hhs_care_lag14',
    'hhs_care_roll7_mean', 'hhs_care_roll14_mean', 'hhs_care_roll7_std',
    'day_of_week', 'month', 'is_weekend'
]
feat_df = df[feature_cols + ['hhs_care']].dropna()
X = feat_df[feature_cols]
y = feat_df['hhs_care']

split = len(feat_df) - TEST_DAYS
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

for ModelClass, name, tag in [
    (RandomForestRegressor(n_estimators=200, random_state=42), "Random Forest", "rf"),
    (GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42), "Gradient Boosting", "gb"),
]:
    m = ModelClass
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    results.append(metrics(y_test, pred, name))
    with open(f"models/{tag}_model.pkl", "wb") as f:
        pickle.dump(m, f)
    print(f"  -> Saved models/{tag}_model.pkl")

# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv("outputs/model_comparison.csv", index=False)
print(f"\nModel Comparison:\n{res_df.to_string(index=False)}")

# ─── FORECAST NEXT 30 DAYS ────────────────────────────────────────────────────
# Using Exponential Smoothing (refit on all data) + GB model
es_full = ExponentialSmoothing(target, trend='add', seasonal='add', seasonal_periods=7).fit()
future_dates = pd.date_range(target.index[-1] + pd.Timedelta(days=1), periods=HORIZON, freq='D')
es_future = es_full.forecast(HORIZON)
es_future.index = future_dates

# Save forecasts
es_future.to_csv("outputs/es_forecast_30d.csv", header=['hhs_care_forecast'])
print(f"\n[+] 30-day ES forecast saved to outputs/es_forecast_30d.csv")

# ─── PLOT: TEST PERIOD COMPARISON ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Model Forecast vs Actual (90-day test period)', fontsize=14, fontweight='bold')

for ax, (pred_series, label, color) in zip(axes.flat, [
    (naive_pred, "Naive", "gray"),
    (es_pred, "Exp Smoothing", "orange"),
    (arima_pred, "ARIMA", "purple"),
    (pd.Series(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42).fit(X_train, y_train).predict(X_test), index=X_test.index), "Gradient Boosting", "steelblue"),
]):
    ax.plot(test.index, test.values, label='Actual', color='black', linewidth=1.5)
    ax.plot(test.index, pred_series.values[:len(test)], label=label, color=color, linewidth=1.5, linestyle='--')
    ax.set_title(f'{label}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] Saved outputs/model_comparison.png")

# ─── PLOT: FUTURE FORECAST ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
last_180 = target.iloc[-180:]
ax.plot(last_180.index, last_180.values, color='steelblue', label='Historical', linewidth=1.5)
ax.plot(es_future.index, es_future.values, color='tomato', linewidth=2, linestyle='--', label='30-day Forecast (ES)')

# Confidence interval (±1.5 std of residuals)
resid_std = (target - es_full.fittedvalues).std()
ax.fill_between(es_future.index,
                es_future.values - 1.5 * resid_std,
                es_future.values + 1.5 * resid_std,
                alpha=0.2, color='tomato', label='90% CI')
ax.axvline(target.index[-1], color='gray', linestyle=':', linewidth=1)
ax.set_title('30-Day Forecast: Children in HHS Care', fontsize=13)
ax.set_ylabel('Children in HHS Care')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/future_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] Saved outputs/future_forecast.png")
print("\nSTEP 2 COMPLETE. Run step3_kpis.py next.")
