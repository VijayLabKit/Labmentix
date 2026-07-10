# Tesla Stock Price Prediction Using Deep Learning (SimpleRNN and LSTM)
# This script trains models and saves them for the Streamlit app

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow:", tf.__version__)
print("All libraries loaded successfully.")

# ============================================================
# 2. LOAD DATASET
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'TSLA.csv')
df = pd.read_csv(DATA_PATH)

print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- Columns ---")
print(df.columns.tolist())
print("\n--- Data Types ---")
print(df.dtypes)
print("\n--- Head ---")
print(df.head())
print("\n--- Tail ---")
print(df.tail())

# ============================================================
# 3. DATA CLEANING
# ============================================================
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Duplicate Records ---")
print(df.duplicated().sum())

df = df.drop_duplicates()
df = df.ffill()

print("\n--- Cleaned Shape ---", df.shape)

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
df['Daily_Return'] = df['Close'].pct_change()
df['MA_20']  = df['Close'].rolling(20).mean()
df['MA_50']  = df['Close'].rolling(50).mean()
df['MA_100'] = df['Close'].rolling(100).mean()
df['Volatility'] = df['Daily_Return'].rolling(20).std()
df.dropna(inplace=True)

print("\n--- After Feature Engineering ---", df.shape)
print(df.head())

# ============================================================
# 5. SCALING & SEQUENCE GENERATION
# ============================================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(df[['Close']])

WINDOW = 60

def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_close, WINDOW)
X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 6. MODEL 1: SimpleRNN
# ============================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def build_rnn(units=64, dropout=0.2, lr=0.001):
    model = Sequential([
        SimpleRNN(units, return_sequences=True, input_shape=(WINDOW, 1)),
        Dropout(dropout),
        SimpleRNN(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss='mse')
    return model

rnn_model = build_rnn()
rnn_model.summary()

rnn_checkpoint = os.path.join(MODELS_DIR, 'rnn_model.h5')
rnn_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(rnn_checkpoint, monitor='val_loss', save_best_only=True)
]

print("\nTraining SimpleRNN...")
rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=80,
    batch_size=32,
    validation_split=0.1,
    callbacks=rnn_callbacks,
    verbose=1
)

# ============================================================
# 7. MODEL 2: LSTM
# ============================================================
def build_lstm(units=64, dropout=0.2, lr=0.001):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(WINDOW, 1)),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss='mse')
    return model

lstm_model = build_lstm()
lstm_model.summary()

lstm_checkpoint = os.path.join(MODELS_DIR, 'lstm_model.h5')
lstm_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(lstm_checkpoint, monitor='val_loss', save_best_only=True)
]

print("\nTraining LSTM...")
lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=80,
    batch_size=32,
    validation_split=0.1,
    callbacks=lstm_callbacks,
    verbose=1
)

# ============================================================
# 8. EVALUATION
# ============================================================
def evaluate_model(model, X_test, y_test, scaler, name):
    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse  = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(actual, pred)
    r2   = r2_score(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    print(f"\n--- {name} Metrics ---")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.4f}%")
    return {"Model": name, "MSE": round(mse,4), "RMSE": round(rmse,4),
            "MAE": round(mae,4), "MAPE": round(mape,4), "R2": round(r2,4)}, pred, actual

rnn_metrics, rnn_pred, actual = evaluate_model(rnn_model, X_test, y_test, scaler, "SimpleRNN")
lstm_metrics, lstm_pred, _    = evaluate_model(lstm_model, X_test, y_test, scaler, "LSTM")

comparison = pd.DataFrame([rnn_metrics, lstm_metrics])
print("\n--- Model Comparison Table ---")
print(comparison.to_string(index=False))

# ============================================================
# 9. FORECASTING FUNCTIONS
# ============================================================
def recursive_forecast(model, last_seq, n_days, scaler):
    seq = last_seq.copy()
    preds = []
    for _ in range(n_days):
        inp = seq[-WINDOW:].reshape(1, WINDOW, 1)
        p = model.predict(inp, verbose=0)[0, 0]
        preds.append(p)
        seq = np.append(seq, p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

last_seq = scaled_close[-WINDOW:, 0]

print("\n--- LSTM Forecasts ---")
for days in [1, 5, 10]:
    fc = recursive_forecast(lstm_model, last_seq, days, scaler)
    print(f"Next {days} day(s): {fc}")

# ============================================================
# 10. SAVE SCALER & METRICS
# ============================================================
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(comparison, os.path.join(MODELS_DIR, 'metrics.pkl'))

# Save training history
import json
hist_data = {
    'rnn_loss': rnn_history.history['loss'],
    'rnn_val_loss': rnn_history.history['val_loss'],
    'lstm_loss': lstm_history.history['loss'],
    'lstm_val_loss': lstm_history.history['val_loss'],
}
with open(os.path.join(MODELS_DIR, 'history.json'), 'w') as f:
    json.dump(hist_data, f)

print("\n✅ All models and artifacts saved successfully.")
print(f"Models saved in: {MODELS_DIR}")
