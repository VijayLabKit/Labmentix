# ⚡ Tesla Stock Price Prediction Using Deep Learning

> Industry-grade end-to-end project comparing SimpleRNN and LSTM for TSLA stock price forecasting.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

Develop a deep learning-based time series forecasting system to predict Tesla's closing stock price using historical OHLCV data. The system compares SimpleRNN and LSTM models to identify the best architecture for financial forecasting.

---

## 📂 Project Structure

```
Tesla_Stock_Prediction/
│
├── data/
│   └── TSLA.csv                        # Historical Tesla stock data
│
├── notebooks/
│   └── Tesla_Stock_Prediction.ipynb    # Full Jupyter notebook
│   └── Tesla_Stock_Prediction.py       # Training script
│
├── models/
│   ├── lstm_model.h5                   # Trained LSTM model
│   ├── rnn_model.h5                    # Trained SimpleRNN model
│   ├── scaler.pkl                      # MinMaxScaler
│   ├── metrics.pkl                     # Model metrics comparison
│   └── history.json                    # Training history
│
├── app/
│   └── streamlit_app.py                # Interactive Streamlit dashboard
│
├── reports/
│   └── Final_Report.pdf                # Detailed project report
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 📊 Dataset

| Column     | Description                        |
|------------|------------------------------------|
| Date       | Trading date                       |
| Open       | Opening price (USD)                |
| High       | Daily high price (USD)             |
| Low        | Daily low price (USD)              |
| Close      | Closing price (USD) — **Target**   |
| Adj Close  | Adjusted closing price             |
| Volume     | Number of shares traded            |

- **Source:** Yahoo Finance / Kaggle
- **Period:** June 2010 – February 2020
- **Records:** 2,416 rows

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/VijayLabKit/Tesla_Stock_Prediction.git
cd Tesla_Stock_Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train Models (from scratch)
```bash
python notebooks/Tesla_Stock_Prediction.py
```

### 2. Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook notebooks/Tesla_Stock_Prediction.ipynb
```

---

## 🤖 Models

### SimpleRNN
```
SimpleRNN(64) → Dropout(0.2) → SimpleRNN(32) → Dropout(0.2) → Dense(32) → Dense(1)
Optimizer: Adam | Loss: MSE | Window: 60 days
```

### LSTM
```
LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(32) → Dense(1)
Optimizer: Adam | Loss: MSE | Window: 60 days
```

---

## 📈 Results

| Model     | MSE       | RMSE    | MAE     | MAPE   | R²     |
|-----------|-----------|---------|---------|--------|--------|
| SimpleRNN | 2181.91   | 46.71   | 28.02   | 7.95%  | 0.602  |
| **LSTM**  | **900.73**| **30.01**| **20.77**| **6.56%** | **0.836** |

**🏆 Winner: LSTM** — significantly outperforms SimpleRNN across all metrics due to its gating mechanism for long-term dependency learning.

---

## 🔮 Forecasting

- **1-Day Forecast:** ~$544.91
- **5-Day Forecast:** Gradual convergence trend
- **10-Day Forecast:** Momentum-based recursive prediction

---

## 🖥️ Streamlit App Features

- 📂 Upload custom CSV data
- 📈 Interactive stock price visualization
- 🔬 Comprehensive EDA (distributions, correlations, volatility)
- 🤖 Real-time predictions from trained models
- 🔮 1/5/10-day future forecasting
- 📊 Model comparison dashboard
- 📥 Download predictions as CSV

---

## 🚀 Streamlit Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file as `app/streamlit_app.py`
4. Deploy!

---

## 🔭 Future Enhancements

| Feature | Description |
|---------|-------------|
| GRU | Gated Recurrent Unit comparison |
| Transformer | Attention-based architecture |
| Sentiment | News + Reddit sentiment features |
| Macro Data | Interest rates, CPI integration |
| Ensemble | Multi-model weighted voting |
| Hyperparameter Tuning | Optuna/Keras Tuner automation |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Past stock performance does not guarantee future results. Do not use these predictions for actual investment decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ishan Chowdhury** — Labmentix · Deep Learning Enthusiast · Cybersecurity & Data Science
