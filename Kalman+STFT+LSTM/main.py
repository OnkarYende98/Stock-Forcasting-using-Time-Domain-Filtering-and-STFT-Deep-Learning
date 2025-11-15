import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from pykalman import KalmanFilter
from datetime import datetime

# =====================================================
# Step 1: Download Data
# =====================================================
TICKER = "^NSEI"
START = "2015-01-01"
END = datetime.today().strftime("%Y-%m-%d")

print(f"📥 Downloading Nifty 50 data up to: {END}")
df = yf.download(TICKER, start=START, end=END)
raw_prices = df["Close"].values.reshape(-1, 1)
print(f"✅ Downloaded {len(raw_prices)} points.")

# =====================================================
# Step 2: Apply Kalman Filter (Smoothing)
# =====================================================
print("\n⚙️ Applying Kalman Filter for smoothing...")

kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=raw_prices[0],
    initial_state_covariance=1,
    observation_covariance=1,   # measurement noise
    transition_covariance=0.01  # process noise
)

state_means, _ = kf.filter(raw_prices)
smoothed_prices = state_means.flatten()

# =====================================================
import numpy as np

def causal_stft_features(price_series, window_size=64, num_freqs=10):
    """
    Compute causal STFT-like FFT features (only past data) for each time step.
    Returns arrays of amplitudes and phases (num_points x num_freqs).
    """
    n = len(price_series)
    fft_amplitudes = np.zeros((n, num_freqs))
    fft_phases = np.zeros((n, num_freqs))

    for i in range(n):
        # Take only past window (causal)
        start = max(0, i - window_size + 1)
        segment = price_series[start:i+1]

        # Skip empty segments
        if len(segment) == 0:
            continue

        # Compute FFT
        fft_vals = np.fft.fft(segment)
        amps = np.abs(fft_vals)[:len(fft_vals)//2]
        phases = np.angle(fft_vals)[:len(fft_vals)//2]

        # If segment smaller than num_freqs, pad with zeros
        if len(amps) < num_freqs:
            pad_len = num_freqs - len(amps)
            amps = np.pad(amps, (0, pad_len))
            phases = np.pad(phases, (0, pad_len))

        # Select strongest frequencies
        top_idx = np.argsort(amps)[-num_freqs:]
        fft_amplitudes[i, :] = amps[top_idx]
        fft_phases[i, :] = phases[top_idx]

    return fft_amplitudes, fft_phases


amp, phase = causal_stft_features(smoothed_prices, window_size=64, num_freqs=10)

# Combine features: [raw, smoothed, mean amplitude, mean phase]
features = np.column_stack([
    raw_prices.flatten(),
    smoothed_prices,
    amp.mean(axis=1),
    phase.mean(axis=1)
])

print("✅ Feature matrix shape:", features.shape)

# =====================================================
# Step 4: Train-Test Split
# =====================================================
split_idx = int(0.8 * len(features))
train_features, test_features = features[:split_idx], features[split_idx:]
train_prices, test_prices = raw_prices[:split_idx], raw_prices[split_idx:]

print(f"📊 Train size: {len(train_features)}, Test size: {len(test_features)}")

# =====================================================
# Step 5: Scaling
# =====================================================
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(train_features)
X_test_scaled = feature_scaler.transform(test_features)
y_train_scaled = target_scaler.fit_transform(train_prices).flatten()
y_test_scaled = target_scaler.transform(test_prices).flatten()

# =====================================================
# Step 6: Create Sequences (No Leakage)
# =====================================================
def create_sequences(features, target, time_steps=60):
    Xs, ys = [], []
    for i in range(len(features) - time_steps):
        Xs.append(features[i:(i + time_steps)])
        ys.append(target[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

print("✅ Sequence data shapes:")
print("Train:", X_train_seq.shape, "Test:", X_test_seq.shape)

# =====================================================
# Step 7: Build & Train LSTM
# =====================================================
print("\n🚀 Training LSTM model...")

model = Sequential([
    LSTM(64, return_sequences=False,
         input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
         kernel_regularizer=l2(0.001),
         recurrent_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

# =====================================================
# Step 8: Predictions
# =====================================================
y_train_pred_scaled = model.predict(X_train_seq, verbose=0)
y_test_pred_scaled = model.predict(X_test_seq, verbose=0)

# Inverse transform
y_train_real = target_scaler.inverse_transform(y_train_seq.reshape(-1, 1)).flatten()
y_train_pred_real = target_scaler.inverse_transform(y_train_pred_scaled).flatten()
y_test_real = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
y_test_pred_real = target_scaler.inverse_transform(y_test_pred_scaled).flatten()

# =====================================================
# Step 9: Metrics
# =====================================================
rmse_train = np.sqrt(mean_squared_error(y_train_real, y_train_pred_real))
rmse_test = np.sqrt(mean_squared_error(y_test_real, y_test_pred_real))
mape_train = mean_absolute_percentage_error(y_train_real, y_train_pred_real) * 100
mape_test = mean_absolute_percentage_error(y_test_real, y_test_pred_real) * 100
r2_train = r2_score(y_train_real, y_train_pred_real)
r2_test = r2_score(y_test_real, y_test_pred_real)

print("\n📈 Model Performance:")
print(f"Train RMSE: {rmse_train:.4f}, Test RMSE: {rmse_test:.4f}")
print(f"Train MAPE: {mape_train:.2f}%, Test MAPE: {mape_test:.2f}%")
print(f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

# =====================================================
# Step 10: Visualization
# =====================================================
plt.figure(figsize=(18, 12))

# Loss curve
plt.subplot(3, 2, 1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()

# Raw vs smoothed
plt.subplot(3, 2, 2)
plt.plot(raw_prices, label="Raw Prices", alpha=0.7)
plt.plot(smoothed_prices, label="Kalman Smoothed")
plt.title("Raw vs Kalman Smoothed Prices")
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
plt.axvline(x=len(y_train_real), color='r', linestyle='--', alpha=0.7, label='Train/Test Split')
plt.plot(full_true, label="True Price")
plt.plot(full_pred, label="Predicted Price")
plt.title("Full Timeline (Train + Test)")
plt.legend()

plt.tight_layout()
plt.show()
