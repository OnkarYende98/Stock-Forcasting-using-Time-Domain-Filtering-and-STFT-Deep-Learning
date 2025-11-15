import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.signal import savgol_coeffs
from datetime import date

# -----------------------------
# Step 1: Download Data
# -----------------------------
TICKER = "^NSEI"
START = "2015-01-01"
END = date.today().strftime("%Y-%m-%d")

df = yf.download(TICKER, start=START, end=END)
raw_prices = df["Close"].values.flatten()
print(f"Downloaded {len(raw_prices)} points until {END}.")

# -----------------------------
# Step 2: Apply Causal Savitzky–Golay Filter
# -----------------------------
def causal_savgol_filter(prices, window_size=7, poly_order=3):
    """Applies a causal (past-only) Savitzky–Golay filter."""
    if window_size % 2 == 0:
        window_size -= 1
    coeffs = savgol_coeffs(window_size, poly_order)
    smoothed = np.zeros_like(prices)
    half_window = window_size - 1

    for i in range(len(prices)):
        start = max(0, i - half_window)
        segment = prices[start:i+1]
        valid_coeffs = coeffs[-len(segment):]
        smoothed[i] = np.dot(valid_coeffs, segment) / valid_coeffs.sum()
    return smoothed

print("Applying causal Savitzky–Golay smoothing...")
sg_prices = causal_savgol_filter(raw_prices, window_size=7, poly_order=3)

# -----------------------------
# Step 3: Compute Global FFT (Single transform on full signal)
# -----------------------------
def compute_global_fft(signal, num_freqs=5):
    """Compute FFT once and return top amplitudes & phases globally."""
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n)

    # Use only positive frequencies
    pos_mask = freqs > 0
    amps = np.abs(fft_vals[pos_mask])
    phases = np.angle(fft_vals[pos_mask])

    # Select top frequency components by amplitude
    top_idx = np.argsort(amps)[-num_freqs:]
    amps_top = amps[top_idx]
    phases_top = phases[top_idx]

    return amps_top, phases_top

amps_top, phases_top = compute_global_fft(sg_prices, num_freqs=5)

# Normalize global FFT features
amps_top = (amps_top - np.min(amps_top)) / (np.max(amps_top) - np.min(amps_top))
phases_top = (phases_top - np.min(phases_top)) / (np.max(phases_top) - np.min(phases_top))

# Combine into a single repeated vector (for all timesteps)
global_fft_features = np.tile(np.concatenate([amps_top, phases_top]), (len(sg_prices), 1))

# -----------------------------
# Step 4: Combine FFT + Savitzky Features
# -----------------------------
sg_prices = sg_prices.reshape(-1, 1)
combined_features = np.hstack([sg_prices, global_fft_features])  # [price, FFT features]
target = raw_prices.reshape(-1, 1)

# -----------------------------
# Step 5: Scaling
# -----------------------------
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_scaled = feature_scaler.fit_transform(combined_features)
target_scaled = target_scaler.fit_transform(target)

# -----------------------------
# Step 6: Create Sequences (No Leakage)
# -----------------------------
def create_sequences(features, target, time_steps=60):
    Xs, ys = [], []
    for i in range(len(features) - time_steps):
        Xs.append(features[i:i+time_steps])
        ys.append(target[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60
X_seq, y_seq = create_sequences(features_scaled, target_scaled, time_steps)

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# -----------------------------
# Step 7: Build & Train LSTM
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(time_steps, X_seq.shape[2]),
         kernel_regularizer='l2', recurrent_regularizer='l2'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop]
)

# -----------------------------
# Step 8: Predictions
# -----------------------------
y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

# -----------------------------
# Step 9: Inverse Transform
# -----------------------------
y_train_real = target_scaler.inverse_transform(y_train).flatten()
y_train_pred_real = target_scaler.inverse_transform(y_train_pred_scaled).flatten()
y_test_real = target_scaler.inverse_transform(y_test).flatten()
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

print(f"\n✅ Metrics:")
print(f"Train RMSE: {rmse_train:.4f}, Test RMSE: {rmse_test:.4f}")
print(f"Train MAPE: {mape_train:.2f}%, Test MAPE: {mape_test:.2f}%")
print(f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

# -----------------------------
# Step 11: Visualization
# -----------------------------
plt.figure(figsize=(16, 10))

# 1. Loss curve
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()

# 2. Training fit
plt.subplot(2, 2, 2)
plt.plot(y_train_real, label="True Train")
plt.plot(y_train_pred_real, label="Predicted Train")
plt.title("Training Fit")
plt.legend()

# 3. Testing fit
plt.subplot(2, 2, 3)
plt.plot(y_test_real, label="True Test")
plt.plot(y_test_pred_real, label="Predicted Test")
plt.title("Testing Fit")
plt.legend()

# 4. Full timeline
plt.subplot(2, 2, 4)
full_true = np.concatenate([y_train_real, y_test_real])
full_pred = np.concatenate([y_train_pred_real, y_test_pred_real])
plt.plot(full_true, label="True Price")
plt.plot(full_pred, label="Predicted Price")
plt.axvline(x=len(y_train_real), color='r', linestyle='--', label='Train/Test Split')
plt.title("Full Timeline (Causal Savitzky-Golay + Global FFT + LSTM)")
plt.legend()

plt.tight_layout()
plt.show()
