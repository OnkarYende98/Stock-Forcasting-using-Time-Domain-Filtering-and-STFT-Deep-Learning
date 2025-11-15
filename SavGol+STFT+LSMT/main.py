# ===============================================
# LSTM + Causal (Savitzky–Golay + STFT) cross validation 80-20
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

# -----------------------------
# Step 1: Download Data
# -----------------------------
TICKER = "^NSEI"
START = "2015-01-01"
END = datetime.today().strftime("%Y-%m-%d")  # today's date

print("Fetching data up to:", END)
print("Step 1: Downloading Nifty 50 data...")
df = yf.download(TICKER, start=START, end=END)
raw_prices = df["Close"].values.flatten()
print(f"Downloaded {len(raw_prices)} points.")

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
split_idx = int(0.8 * len(raw_prices))
train_prices = raw_prices[:split_idx]
test_prices = raw_prices[split_idx:]
print(f"Train size: {len(train_prices)}, Test size: {len(test_prices)}")

# -----------------------------
# Step 3: Apply Causal Savitzky–Golay Filter
# -----------------------------
def causal_savgol(prices, window_size=7, poly_order=3):
    smoothed = np.zeros_like(prices)
    for i in range(len(prices)):
        start = max(0, i - window_size + 1)
        end = i + 1
        segment = prices[start:end]
        if len(segment) <= poly_order:
            smoothed[i] = segment[-1]
        else:
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, poly_order)
            smoothed[i] = np.polyval(coeffs, len(segment)-1)
    return smoothed

print("Applying Causal Savitzky–Golay smoothing...")
smoothed_train = causal_savgol(train_prices, window_size=7, poly_order=3)
smoothed_test = causal_savgol(test_prices, window_size=7, poly_order=3)

# -----------------------------
# Step 4: Compute Causal STFT Features
# -----------------------------
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

amp_train, phase_train = compute_causal_stft_features(smoothed_train)
amp_test, phase_test = compute_causal_stft_features(smoothed_test)

# -----------------------------
# Step 5: Build Feature Set
# -----------------------------
features_train = np.column_stack([train_prices, smoothed_train, amp_train, phase_train])
features_test = np.column_stack([test_prices, smoothed_test, amp_test, phase_test])
print("Train features shape:", features_train.shape)
print("Test features shape:", features_test.shape)

# -----------------------------
# Step 6: Scaling
# -----------------------------
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_train_scaled = feature_scaler.fit_transform(features_train[:, 1:])  # exclude raw price
features_test_scaled = feature_scaler.transform(features_test[:, 1:])
target_train_scaled = target_scaler.fit_transform(train_prices.reshape(-1, 1))
target_test_scaled = target_scaler.transform(test_prices.reshape(-1, 1))

# -----------------------------
# Step 7: Sequence Creation
# -----------------------------
def create_sequences(features, target, time_steps=60):
    Xs, ys = [], []
    for i in range(len(features) - time_steps):
        Xs.append(features[i:i+time_steps])
        ys.append(target[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60
X_train, y_train = create_sequences(features_train_scaled, target_train_scaled, time_steps)
X_test, y_test = create_sequences(features_test_scaled, target_test_scaled, time_steps)
print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

# -----------------------------
# Step 8: Build LSTM Model
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2]),
         kernel_regularizer='l2', recurrent_regularizer='l2'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

# -----------------------------
# Step 9: Predictions
# -----------------------------
y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

y_train_real = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_train_pred_real = target_scaler.inverse_transform(y_train_pred_scaled).flatten()
y_test_real = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_test_pred_real = target_scaler.inverse_transform(y_test_pred_scaled).flatten()

# -----------------------------
# Step 10: Metrics
# -----------------------------
rmse_train = np.sqrt(mean_squared_error(y_train_real, y_train_pred_real))
rmse_test = np.sqrt(mean_squared_error(y_test_real, y_test_pred_real))
mape_train = mean_absolute_percentage_error(y_train_real, y_train_pred_real) * 100
mape_test = mean_absolute_percentage_error(y_test_real, y_test_pred_real) * 100
r2_train = r2_score(y_train_real, y_train_pred_real)
r2_test = r2_score(y_test_real, y_test_pred_real)

print(f"Train RMSE: {rmse_train:.4f}, Test RMSE: {rmse_test:.4f}")
print(f"Train MAPE: {mape_train:.2f}%, Test MAPE: {mape_test:.2f}%")
print(f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

# -----------------------------
# Step 11: Visualization
# -----------------------------
plt.figure(figsize=(16, 12))

# Loss curve
plt.subplot(3, 2, 1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()

# Raw vs Smoothed
plt.subplot(3, 2, 2)
plt.plot(train_prices, label="Train Raw", alpha=0.7)
plt.plot(smoothed_train, label="Train Smoothed")
plt.plot(np.arange(len(train_prices), len(raw_prices)), test_prices, label="Test Raw", alpha=0.7)
plt.plot(np.arange(len(train_prices), len(raw_prices)), smoothed_test, label="Test Smoothed")
plt.title("Raw vs Smoothed Prices")
plt.legend()

# Training fit
plt.subplot(3, 2, 3)
plt.plot(y_train_real, label="True Train")
plt.plot(y_train_pred_real, label="Predicted Train")
plt.title("Training Fit")
plt.legend()

# Testing fit
plt.subplot(3, 2, 4)
plt.plot(y_test_real, label="True Test")
plt.plot(y_test_pred_real, label="Predicted Test")
plt.title("Testing Fit")
plt.legend()

# Full timeline
plt.subplot(3, 2, 5)
full_true = np.concatenate([y_train_real, y_test_real])
full_pred = np.concatenate([y_train_pred_real, y_test_pred_real])
plt.plot(full_true, label="True Price")
plt.plot(full_pred, label="Predicted Price")
plt.axvline(x=len(y_train_real), color='r', linestyle='--', label='Train/Test Split')
plt.title("Full Timeline")
plt.legend()

# STFT visualization (train)
plt.subplot(3, 2, 6)
f, t, Zxx = stft(smoothed_train, nperseg=64, noverlap=32)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude (Train)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [steps]')
plt.colorbar()

plt.tight_layout()
plt.show()
