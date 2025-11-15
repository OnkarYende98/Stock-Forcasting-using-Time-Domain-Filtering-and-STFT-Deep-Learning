Comparative Study of Time Series Forecasting Using LSTM with Different Preprocessing Techniques
Overview

This repository presents a comparative study of deep learning-based stock price forecasting using Long Short-Term Memory (LSTM) networks combined with various time-domain filtering and frequency-domain transformation techniques. The study investigates how causal filtering and spectral features affect predictive performance and generalization for financial time series.

The following preprocessing pipelines are implemented and evaluated:

Causal Savitzky–Golay Filter + Short-Time Fourier Transform (STFT) + LSTM

STFT + LSTM

Causal Savitzky–Golay Filter + Fast Fourier Transform (FFT) + LSTM

Kalman Filter + STFT + LSTM

The evaluation metrics include Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R² score on both training and test datasets. Additionally, 30-day forecasts beyond historical data are generated to assess future predictive capability.

Methodology
1. Causal Filtering

Savitzky–Golay (SAVGOL) Filter:
A polynomial smoothing filter is applied in a causal manner (using only current and past values) to reduce noise while preserving local trends and peaks in the stock price series.

Kalman Filter:
A recursive optimal estimator that smooths the time series by considering both process and measurement noise, enabling adaptive noise reduction in financial signals.

2. Frequency-Domain Feature Extraction

Short-Time Fourier Transform (STFT):
Captures local time-frequency variations by sliding a fixed-size window over the time series. Both amplitude and phase information are extracted as features for the LSTM input.

Fast Fourier Transform (FFT):
Provides global frequency-domain information of the entire series. Peaks in the FFT spectrum highlight dominant periodicities and trends useful for forecasting.

3. LSTM Network

A deep LSTM network is employed to model temporal dependencies in the processed time series.

Architecture includes LSTM layers with dropout regularization and a dense output layer for predicting future stock prices.

The model is trained on sequences of historical data with multivariate features, including raw prices, smoothed values, and frequency-domain features.

Results

| Model                | Train RMSE | Test RMSE | Train MAPE | Test MAPE | Train R² | Test R² |
| -------------------- | ---------- | --------- | ---------- | --------- | -------- | ------- |
| SAVGOL + STFT + LSTM | 233.16     | 317.28    | 1.40%      | 1.12%     | 0.9958   | 0.9484  |
| STFT + LSTM          | 183.93     | 337.64    | 1.11%      | 1.18%     | 0.9974   | 0.9637  |
| SAVGOL + FFT + LSTM  | 314.76     | 427.66    | 1.83%      | 1.54%     | 0.9926   | 0.9410  |
| Kalman + STFT + LSTM | 232.08     | 382.59    | 1.35%      | 1.31%     | 0.9959   | 0.9186  |


 The following models are trained on the Nifty 50 (NSEI^) dataset from 2015 to EOD.
 The user gets the compatibility of working with multiple stocks and indexs by entering the ticker name of the security.

 Observations:

Frequency-domain features (STFT/FFT) generally improve predictive accuracy by providing temporal-frequency patterns not visible in raw prices.

Causal filtering methods reduce noise without introducing lookahead bias, ensuring realistic forecasting.

The STFT-based LSTM model achieved the best generalisation on the test set in terms of R², highlighting the effectiveness of short-term frequency-domain features.

Kalman filtering provides adaptive smoothing but may slightly underperform compared to causal SAVGOL in combination with STFT.

