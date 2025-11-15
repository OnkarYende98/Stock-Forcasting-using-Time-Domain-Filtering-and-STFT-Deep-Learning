# ===============================================
# LSTM + Causal (Savitzky–Golay + STFT) + 30-Day Forecast
# ===============================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.signal import stft
from datetime import datetime

# -----------------------------------------------
# Step 1: User Input
# -----------------------------------------------
TICKER = input("Enter Stock Symbol (e.g. RELIANCE.NS").strip()
START = "2015-01-01"
END = "2025-10-14"

print(f"\nFetching {TICKER} data from {START} to {END} ...")
df = yf.download(TICKER, start=START, end=END)
if df.empty:
    raise ValueError("❌ No data found. Check the stock symbol and try again.")
raw_prices = df["Close"].values.flatten()
print(f"✅ Downloaded {len(raw_prices)} data points.")

# -----------------------------------------------
# Step 2: Train-Test Split
# -----------------------------------------------
split_idx = int(0.8 * len(raw_prices))
train_prices = raw_prices[:split_idx]
test_prices = raw_prices[split_idx:]
print(f"Train size: {len(train_prices)}, Test size: {len(test_prices)}")

# -----------------------------------------------
# Step 3: Causal Savitzky–Golay Filter
# -----------------------------------------------
def causal_savgol(prices, window_size=7, poly_order=3):
    smoothed = np.zeros_like(prices)
    for i in range(len(prices)):
        start = max(0, i - window_size + 1)
        segment = prices[start:i+1]
        if len(segment) <= poly_order:
            smoothed[i] = segment[-1]
        else:
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, poly_order)
            smoothed[i] = np.polyval(coeffs, len(segment)-1)
    return smoothed

print("Applying Causal Savitzky–Golay smoothing...")
smoothed_train = causal_savgol(train_prices)
smoothed_test = causal_savgol(test_prices)

# -----------------------------------------------
# Step 4: Compute STFT Features
# -----------------------------------------------
def compute_causal_stft_features(prices, window_size=128, step_size=1):
    n = len(prices)
    amp_features = np.zeros(n)
    phase_features = np.zeros(n)

    for t in range(window_size, n, step_size):
        segment = prices[t-window_size:t]
        f, _, Zxx = stft(segment, nperseg=64, noverlap=32)
        amplitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        amp_features[t] = np.mean(amplitude)
        phase_features[t] = np.mean(phase)

    amp_features[:window_size] = amp_features[window_size]
    phase_features[:window_size] = phase_features[window_size]
    return amp_features, phase_features

print("Computing STFT features...")
amp_train, phase_train = compute_causal_stft_features(smoothed_train)
amp_test, phase_test = compute_causal_stft_features(smoothed_test)

# -----------------------------------------------
# Step 5: Feature Stacking
# -----------------------------------------------
features_train = np.column_stack([train_prices, smoothed_train, amp_train, phase_train])
features_test = np.column_stack([test_prices, smoothed_test, amp_test, phase_test])

# -----------------------------------------------
# Step 6: Scaling
# -----------------------------------------------
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_train_scaled = feature_scaler.fit_transform(features_train[:, 1:])
features_test_scaled = feature_scaler.transform(features_test[:, 1:])
target_train_scaled = target_scaler.fit_transform(train_prices.reshape(-1, 1))
target_test_scaled = target_scaler.transform(test_prices.reshape(-1, 1))

# -----------------------------------------------
# Step 7: Sequence Preparation
# -----------------------------------------------
def create_sequences(features, target, time_steps=60):
    Xs, ys = [], []
    for i in range(len(features) - time_steps):
        Xs.append(features[i:i+time_steps])
        ys.append(target[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60
X_train, y_train = create_sequences(features_train_scaled, target_train_scaled, time_steps)
X_test, y_test = create_sequences(features_test_scaled, target_test_scaled, time_steps)

# -----------------------------------------------
# Step 8: Build & Train LSTM
# -----------------------------------------------
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer='l2', recurrent_regularizer='l2'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nTraining LSTM model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

# -----------------------------------------------
# Step 9: Evaluate & Predict
# -----------------------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_real = target_scaler.inverse_transform(y_train)
y_test_real = target_scaler.inverse_transform(y_test)
y_train_pred_real = target_scaler.inverse_transform(y_train_pred)
y_test_pred_real = target_scaler.inverse_transform(y_test_pred)

print("\nPerformance Metrics:")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train_real, y_train_pred_real)):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test_real, y_test_pred_real)):.4f}")
print(f"Test MAPE: {mean_absolute_percentage_error(y_test_real, y_test_pred_real)*100:.2f}%")

# -----------------------------------------------
# Step 10: 30-Day Future Forecast
# -----------------------------------------------
def forecast_future(model, last_window, n_future, feature_scaler, target_scaler):
    future_predictions = []
    current_input = last_window.copy()

    for _ in range(n_future):
        pred_scaled = model.predict(current_input[np.newaxis, :, :])
        pred_real = target_scaler.inverse_transform(pred_scaled)[0, 0]
        future_predictions.append(pred_real)

        # create new feature row (dummy except price)
        next_features = np.concatenate([
            current_input[1:],  # remove first
            np.expand_dims(current_input[-1], axis=0)
        ])
        # update first feature (price-like feature)
        next_features[-1, 0] = feature_scaler.transform([[pred_real, 0, 0]])[0, 0]
        current_input = next_features

    return np.array(future_predictions)

print("\nGenerating 30-Day Forecast...")
n_future = 30
last_window = X_test[-1]
future_pred = forecast_future(model, last_window, n_future, feature_scaler, target_scaler)
print(f"✅ Forecasted next {n_future} days.")

# -----------------------------------------------
# Step 11: Visualization
# -----------------------------------------------
plt.figure(figsize=(14, 7))
plt.plot(raw_prices, label="Historical Prices")
future_days = np.arange(len(raw_prices), len(raw_prices) + n_future)
plt.plot(future_days, future_pred, label="Predicted Future (Next 30 Days)", linestyle="--", color='r')
plt.axvline(x=len(raw_prices)-len(test_prices), color='gray', linestyle='--', label='Train/Test Split')
plt.title(f"30-Day Price Forecast for {TICKER}")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.show()



# -----------------------------------------------
# Step 11: Visualization (Actual vs Predicted + Future Forecast)
# -----------------------------------------------

plt.figure(figsize=(15, 8))

# 1) PLOT TRAIN + TEST ACTUALS
plt.plot(raw_prices, label="Actual Price", color="black", linewidth=2)

# 2) PLOT TEST SET PREDICTIONS (aligned correctly)
test_start_index = len(train_prices)
test_pred_plot = np.empty(len(raw_prices))
test_pred_plot[:] = np.nan
test_pred_plot[test_start_index + time_steps : test_start_index + time_steps + len(y_test_pred_real)] = y_test_pred_real.flatten()

plt.plot(test_pred_plot, label="Predicted (Test Data)", color="blue", linewidth=2)

# 3) PLOT FUTURE FORECAST
future_x = np.arange(len(raw_prices), len(raw_prices) + n_future)
plt.plot(future_x, future_pred, label="30-Day Forecast", linestyle="--", color="red", linewidth=2)

# 4) TRAIN / TEST SPLIT LINE
plt.axvline(x=test_start_index, color='gray', linestyle='--', label='Train/Test Split')

plt.title(f"{TICKER} — Actual vs Predicted (Test) + 30-Day Forecast", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Stock Price", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------------------------
# Step 12: Final Output
# -----------------------------------------------
print("\n📊 Next 30-Day Predicted Prices:")
for i, p in enumerate(future_pred, 1):
    print(f"Day {i}: {p:.2f}")
