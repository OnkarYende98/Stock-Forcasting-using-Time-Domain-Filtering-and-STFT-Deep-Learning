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
END   = datetime.today().strftime("%Y-%m-%d")  # today's date

print("Step 1: Downloading Nifty 50 data...")
df = yf.download(TICKER, start=START, end=END)
raw_prices = df["Close"].values.flatten()
print(f"Downloaded {len(raw_prices)} points.")

# -----------------------------
# Step 2: Causal STFT Feature Extraction
# -----------------------------
print("Step 2: Computing causal STFT features...")

def causal_stft_features(prices, window_size=128, nperseg=64, noverlap=32):
    n = len(prices)
    amp = np.zeros(n)
    phase = np.zeros(n)

    for t in range(window_size, n):
        seg = prices[t-window_size:t]  # only past data
        f, _, Zxx = stft(seg, nperseg=nperseg, noverlap=noverlap)
        amp[t] = np.mean(np.abs(Zxx))
        phase[t] = np.mean(np.angle(Zxx))

    # fill initial points with first computed value
    amp[:window_size] = amp[window_size]
    phase[:window_size] = phase[window_size]

    return amp, phase

amp_resized, phase_resized = causal_stft_features(raw_prices)

# Stack features: [Price, Amplitude, Phase]
features = np.column_stack([raw_prices, amp_resized, phase_resized])
print("Causal STFT features shape:", features.shape)

# -----------------------------
# Step 3: Scaling
# -----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)
y_scaled = X_scaled[:, 0]  # target is raw price

# -----------------------------
# Step 4: Sequence Creation
# -----------------------------
def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
print("Train/Test sizes:", X_train.shape, X_test.shape)

# -----------------------------
# Step 5: LSTM Model
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
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
# Step 6: Predictions
# -----------------------------
y_train_pred_scaled = model.predict(X_train, verbose=0)
y_test_pred_scaled  = model.predict(X_test, verbose=0)

def inverse_price_transform(y_scaled, X_template, scaler):
    y_full = np.zeros((len(y_scaled), X_template.shape[2]))
    y_full[:, 0] = y_scaled.flatten()
    return scaler.inverse_transform(y_full)[:, 0]

y_train_real = inverse_price_transform(y_train, X_train, scaler)
y_train_pred_real = inverse_price_transform(y_train_pred_scaled, X_train, scaler)
y_test_real = inverse_price_transform(y_test, X_test, scaler)
y_test_pred_real = inverse_price_transform(y_test_pred_scaled, X_test, scaler)

# -----------------------------
# Step 7: Metrics
# -----------------------------
rmse_train = np.sqrt(mean_squared_error(y_train_real, y_train_pred_real))
rmse_test  = np.sqrt(mean_squared_error(y_test_real, y_test_pred_real))
mape_train = mean_absolute_percentage_error(y_train_real, y_train_pred_real) * 100
mape_test  = mean_absolute_percentage_error(y_test_real, y_test_pred_real) * 100
r2_train = r2_score(y_train_real, y_train_pred_real)
r2_test  = r2_score(y_test_real, y_test_pred_real)

print(f"Train RMSE: {rmse_train:.4f}, Test RMSE: {rmse_test:.4f}")
print(f"Train MAPE: {mape_train:.2f}%, Test MAPE: {mape_test:.2f}%")
print(f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

# -----------------------------
# Step 8: Visualization
# -----------------------------
plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Curve"); plt.xlabel("Epochs"); plt.ylabel("MSE"); plt.legend()

plt.subplot(2,2,2)
plt.plot(y_train_real, label="True Train")
plt.plot(y_train_pred_real, label="Predicted Train")
plt.title("Training Fit"); plt.legend()

plt.subplot(2,2,3)
plt.plot(y_test_real, label="True Test")
plt.plot(y_test_pred_real, label="Predicted Test")
plt.title("Testing Fit"); plt.legend()

plt.subplot(2,2,4)
full_true = np.concatenate([y_train_real, y_test_real])
full_pred = np.concatenate([y_train_pred_real, y_test_pred_real])
plt.axvline(x=len(y_train_real), color='r', linestyle='--', alpha=0.7)
plt.plot(full_true, label="True Price")
plt.plot(full_pred, label="Predicted Price")
plt.title("Full Timeline (Train+Test)"); plt.legend()
plt.tight_layout()
plt.show()
