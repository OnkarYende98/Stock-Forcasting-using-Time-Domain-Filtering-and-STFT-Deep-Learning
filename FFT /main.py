import numpy as np

# ---- Sample Signal ----
# Example: a noisy signal composed of 2 sine waves
fs = 1000  # sampling frequency
t = np.linspace(0, 1, fs)
signal = 3*np.sin(2*np.pi*50*t) + 2*np.sin(2*np.pi*120*t) + np.random.randn(len(t))

# ---- FFT ----
N = len(signal)
freqs = np.fft.fftfreq(N, 1/fs)
fft_vals = np.fft.fft(signal)

# Amplitude spectrum
amplitude = np.abs(fft_vals) / N

# Phase spectrum
phase = np.angle(fft_vals)

print("Frequencies:", freqs[:10])
print("Amplitude:", amplitude[:10])
print("Phase:", phase[:10])
