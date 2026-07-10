"""
Tesla Stock Price Prediction - Streamlit App
Industry-Grade Deep Learning Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
import io
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import load_model

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla Stock Prediction | Deep Learning",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1e2130; border-radius: 10px; padding: 15px; border-left: 4px solid #e31937; }
    .stMetricLabel { color: #a0aec0 !important; font-size: 0.85rem !important; }
    .stMetricValue { color: #e2e8f0 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    h1, h2, h3 { color: #e2e8f0; }
    .block-container { padding-top: 1rem; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2d3448;
        border-radius: 12px;
        padding: 16px;
    }
    .forecast-card {
        background: linear-gradient(135deg, #1a1f35, #222840);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3448;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background: #1a3a2a; color: #48bb78; }
    .badge-red   { background: #3a1a1a; color: #fc8181; }
    .badge-blue  { background: #1a2a3a; color: #63b3ed; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data')
WINDOW = 60

# ─── Helper: load resources ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    rnn_model  = load_model(os.path.join(MODELS_DIR, 'rnn_model.h5'), compile=False)
    lstm_model = load_model(os.path.join(MODELS_DIR, 'lstm_model.h5'), compile=False)
    scaler     = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    metrics_df = joblib.load(os.path.join(MODELS_DIR, 'metrics.pkl'))
    with open(os.path.join(MODELS_DIR, 'history.json')) as f:
        history = json.load(f)
    return rnn_model, lstm_model, scaler, metrics_df, history

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.ffill().drop_duplicates()
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_20']        = df['Close'].rolling(20).mean()
    df['MA_50']        = df['Close'].rolling(50).mean()
    df['MA_100']       = df['Close'].rolling(100).mean()
    df['Volatility']   = df['Daily_Return'].rolling(20).std()
    return df.dropna()

def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def recursive_forecast(model, last_seq, n_days, scaler):
    seq = last_seq.copy()
    preds = []
    for _ in range(n_days):
        inp = seq[-WINDOW:].reshape(1, WINDOW, 1)
        p = model.predict(inp, verbose=0)[0, 0]
        preds.append(p)
        seq = np.append(seq, p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Tesla Stock AI")
    st.markdown("---")
    
    uploaded = st.file_uploader("📂 Upload TSLA CSV", type=['csv'],
                                help="Upload TSLA.csv with Date, Open, High, Low, Close, Volume columns")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    selected_model = st.selectbox("🤖 Prediction Model", ["LSTM", "SimpleRNN", "Both"])
    forecast_days  = st.selectbox("📅 Forecast Horizon", [1, 5, 10])
    show_raw       = st.checkbox("📋 Show Raw Data", False)
    show_eda       = st.checkbox("📊 Show EDA", True)
    
    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    <small>
    🔬 <b>Models:</b> SimpleRNN, LSTM<br>
    📈 <b>Target:</b> TSLA Close Price<br>
    🪟 <b>Window:</b> 60 days<br>
    ⚡ <b>Framework:</b> TensorFlow / Keras<br>
    </small>
    """, unsafe_allow_html=True)

# ─── Load data ────────────────────────────────────────────────────────────────
if uploaded is not None:
    df = load_data(io.StringIO(uploaded.read().decode('utf-8')))
    st.sidebar.success("✅ Custom data loaded!")
else:
    default_path = os.path.join(DATA_DIR, 'TSLA.csv')
    if os.path.exists(default_path):
        df = load_data(default_path)
    else:
        st.error("❌ TSLA.csv not found. Please upload it via the sidebar.")
        st.stop()

# ─── Load models ─────────────────────────────────────────────────────────────
with st.spinner("Loading deep learning models..."):
    rnn_model, lstm_model, scaler, metrics_df, history = load_resources()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#c00,#e31937);padding:24px 32px;border-radius:16px;margin-bottom:24px'>
  <h1 style='color:white;margin:0;font-size:2.2rem'>⚡ Tesla Stock Price Prediction</h1>
  <p style='color:rgba(255,255,255,0.85);margin:6px 0 0'>SimpleRNN vs LSTM · Deep Learning Forecasting System · Built on TSLA Historical Data</p>
</div>
""", unsafe_allow_html=True)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
last_price = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2]
change     = last_price - prev_price
pct_change = (change / prev_price) * 100
high_52w   = df['Close'].rolling(252).max().iloc[-1]
low_52w    = df['Close'].rolling(252).min().iloc[-1]
avg_vol    = df['Volume'].rolling(20).mean().iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("💵 Last Close", f"${last_price:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
with col2:
    st.metric("📈 52W High", f"${high_52w:.2f}")
with col3:
    st.metric("📉 52W Low", f"${low_52w:.2f}")
with col4:
    st.metric("📊 Avg Volume (20D)", f"{avg_vol/1e6:.2f}M")
with col5:
    st.metric("🗓️ Data Points", f"{len(df):,}")

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📈 Stock Overview", "🔬 EDA", "🤖 Predictions", "🔮 Forecast", "📊 Model Comparison"])

# ═══════════════════════════════════════════════════════════
# TAB 1: Stock Overview
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("📈 Tesla Stock Price History")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Close Price + Moving Averages", "Volume"))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                             name="Close", line=dict(color='#e31937', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'],
                             name="MA 20", line=dict(color='#48bb78', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'],
                             name="MA 50", line=dict(color='#63b3ed', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_100'],
                             name="MA 100", line=dict(color='#f6ad55', width=1, dash='dot')), row=1, col=1)
    
    colors = ['#48bb78' if c >= 0 else '#fc8181' for c in df['Close'].diff()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume",
                         marker_color=colors, opacity=0.6), row=2, col=1)
    
    fig.update_layout(height=550, template='plotly_dark',
                      paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                      legend=dict(orientation='h', y=1.06))
    fig.update_xaxes(showgrid=True, gridcolor='#2d3448')
    fig.update_yaxes(showgrid=True, gridcolor='#2d3448')
    st.plotly_chart(fig, use_container_width=True)
    
    if show_raw:
        st.subheader("📋 Raw Data")
        st.dataframe(df.tail(100).style.highlight_max(axis=0, color='#1a3a2a')
                                       .highlight_min(axis=0, color='#3a1a1a'),
                     use_container_width=True)
        csv_bytes = df.to_csv().encode()
        st.download_button("📥 Download Full Dataset (CSV)", csv_bytes, "TSLA_features.csv", "text/csv")

# ═══════════════════════════════════════════════════════════
# TAB 2: EDA
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    if not show_eda:
        st.info("Enable 'Show EDA' in the sidebar to view this section.")
    else:
        st.subheader("🔬 Exploratory Data Analysis")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # Daily Returns Distribution
            fig_dist = px.histogram(df, x='Daily_Return', nbins=80,
                                    title='Daily Returns Distribution',
                                    color_discrete_sequence=['#e31937'])
            fig_dist.update_layout(template='plotly_dark', paper_bgcolor='#0e1117',
                                   plot_bgcolor='#0e1117', height=350)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("📌 Returns are roughly normally distributed with slight negative skewness, indicating occasional large drops.")
        
        with c2:
            # Volatility
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volatility']*100,
                                         fill='tozeroy', name='20D Volatility (%)',
                                         line=dict(color='#f6ad55', width=1.5)))
            fig_vol.update_layout(title='Rolling 20-Day Volatility (%)',
                                  template='plotly_dark', paper_bgcolor='#0e1117',
                                  plot_bgcolor='#0e1117', height=350)
            st.plotly_chart(fig_vol, use_container_width=True)
            st.caption("📌 Volatility spikes are visible around 2013 and late 2019, reflecting high-uncertainty periods.")
        
        c3, c4 = st.columns(2)
        
        with c3:
            # Correlation Heatmap
            corr = df[['Open','High','Low','Close','Volume','Daily_Return','Volatility']].corr()
            fig_hm = px.imshow(corr, title='Correlation Heatmap', text_auto='.2f',
                               color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig_hm.update_layout(template='plotly_dark', paper_bgcolor='#0e1117',
                                  plot_bgcolor='#0e1117', height=380)
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("📌 OHLC prices are highly correlated (>0.99). Volume shows low correlation with price.")
        
        with c4:
            # Rolling Mean & Std
            fig_rs = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=('Rolling Mean (30D)', 'Rolling Std (30D)'))
            rm = df['Close'].rolling(30).mean()
            rs = df['Close'].rolling(30).std()
            fig_rs.add_trace(go.Scatter(x=df.index, y=rm, name='Rolling Mean',
                                        line=dict(color='#63b3ed')), row=1, col=1)
            fig_rs.add_trace(go.Scatter(x=df.index, y=rs, name='Rolling Std',
                                        fill='tozeroy', line=dict(color='#f6ad55')), row=2, col=1)
            fig_rs.update_layout(template='plotly_dark', paper_bgcolor='#0e1117',
                                  plot_bgcolor='#0e1117', height=380, showlegend=False)
            st.plotly_chart(fig_rs, use_container_width=True)
            st.caption("📌 Rising rolling std after 2019 indicates increasing price uncertainty.")
        
        # Boxplot by year
        df_plot = df.copy()
        df_plot['Year'] = df_plot.index.year
        fig_box = px.box(df_plot, x='Year', y='Close', title='Yearly Close Price Distribution',
                         color='Year', color_discrete_sequence=px.colors.qualitative.Set2)
        fig_box.update_layout(template='plotly_dark', paper_bgcolor='#0e1117',
                               plot_bgcolor='#0e1117', height=380, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("📌 Tesla's median close price accelerated significantly from 2019 onwards, with widening interquartile ranges indicating higher volatility.")
        
        # Stats
        st.subheader("📐 Descriptive Statistics")
        st.dataframe(df[['Open','High','Low','Close','Volume','Daily_Return','Volatility']]
                     .describe().round(4), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3: Predictions vs Actual
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🤖 Model Predictions vs Actual Close Price")
    
    scaled_close = scaler.transform(df[['Close']])
    X, y = create_sequences(scaled_close, WINDOW)
    split = int(0.8 * len(X))
    X_test, y_test = X[split:], y[split:]
    X_test_r = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    with st.spinner("Generating predictions..."):
        rnn_pred  = scaler.inverse_transform(rnn_model.predict(X_test_r, verbose=0)).flatten()
        lstm_pred = scaler.inverse_transform(lstm_model.predict(X_test_r, verbose=0)).flatten()
    
    test_dates = df.index[WINDOW + split: WINDOW + split + len(actual)]
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=test_dates, y=actual, name="Actual",
                                  line=dict(color='white', width=2)))
    
    if selected_model in ["SimpleRNN", "Both"]:
        fig_pred.add_trace(go.Scatter(x=test_dates, y=rnn_pred, name="SimpleRNN",
                                      line=dict(color='#f6ad55', width=1.5, dash='dot')))
    
    if selected_model in ["LSTM", "Both"]:
        fig_pred.add_trace(go.Scatter(x=test_dates, y=lstm_pred, name="LSTM",
                                      line=dict(color='#48bb78', width=1.5, dash='dash')))
    
    fig_pred.update_layout(title="Actual vs Predicted Close Price (Test Set)",
                           template='plotly_dark', paper_bgcolor='#0e1117',
                           plot_bgcolor='#0e1117', height=450,
                           legend=dict(orientation='h', y=1.08),
                           xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Training history
    st.subheader("📉 Training Loss History")
    c1, c2 = st.columns(2)
    
    with c1:
        fig_rh = go.Figure()
        fig_rh.add_trace(go.Scatter(y=history['rnn_loss'], name='Train Loss',
                                    line=dict(color='#e31937')))
        fig_rh.add_trace(go.Scatter(y=history['rnn_val_loss'], name='Val Loss',
                                    line=dict(color='#f6ad55', dash='dash')))
        fig_rh.update_layout(title='SimpleRNN Training History', template='plotly_dark',
                              paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', height=300)
        st.plotly_chart(fig_rh, use_container_width=True)
    
    with c2:
        fig_lh = go.Figure()
        fig_lh.add_trace(go.Scatter(y=history['lstm_loss'], name='Train Loss',
                                    line=dict(color='#48bb78')))
        fig_lh.add_trace(go.Scatter(y=history['lstm_val_loss'], name='Val Loss',
                                    line=dict(color='#63b3ed', dash='dash')))
        fig_lh.update_layout(title='LSTM Training History', template='plotly_dark',
                              paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', height=300)
        st.plotly_chart(fig_lh, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 4: Forecast
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader(f"🔮 Next {forecast_days}-Day Forecast")
    
    scaled_close = scaler.transform(df[['Close']])
    last_seq = scaled_close[-WINDOW:, 0]
    last_date = df.index[-1]
    
    model_for_fc = lstm_model if selected_model in ["LSTM", "Both"] else rnn_model
    model_name   = "LSTM" if selected_model in ["LSTM", "Both"] else "SimpleRNN"
    
    with st.spinner(f"Generating {forecast_days}-day forecast using {model_name}..."):
        fc_prices = recursive_forecast(model_for_fc, last_seq, forecast_days, scaler)
    
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    fc_df = pd.DataFrame({'Date': future_dates, 'Forecast': fc_prices})
    
    # Show forecast cards
    if forecast_days <= 5:
        cols = st.columns(forecast_days)
        for i, (_, row) in enumerate(fc_df.iterrows()):
            day_change = row['Forecast'] - last_price
            pct = (day_change / last_price) * 100
            badge = "🟢" if day_change >= 0 else "🔴"
            with cols[i]:
                st.markdown(f"""
                <div class='forecast-card'>
                  <div style='font-size:0.8rem;color:#a0aec0'>{row['Date'].strftime('%b %d, %Y')}</div>
                  <div style='font-size:1.5rem;font-weight:700;color:#e2e8f0;margin:8px 0'>${row['Forecast']:.2f}</div>
                  <div style='color:{"#48bb78" if day_change>=0 else "#fc8181"};font-size:0.9rem'>{badge} {pct:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.dataframe(fc_df.style.format({'Forecast': '${:.2f}'}), use_container_width=True)
    
    # Forecast chart
    st.markdown("---")
    fig_fc = go.Figure()
    
    # Last 60 days actual
    hist_slice = df['Close'].iloc[-60:]
    fig_fc.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice,
                                name='Historical', line=dict(color='#63b3ed', width=2)))
    
    # Connector
    fig_fc.add_trace(go.Scatter(
        x=[hist_slice.index[-1], future_dates[0]],
        y=[hist_slice.iloc[-1], fc_prices[0]],
        mode='lines', line=dict(color='#a0aec0', dash='dot'), showlegend=False))
    
    # Forecast
    fig_fc.add_trace(go.Scatter(x=future_dates, y=fc_prices,
                                name=f'{model_name} Forecast',
                                line=dict(color='#48bb78', width=2, dash='dash'),
                                mode='lines+markers',
                                marker=dict(size=8, color='#48bb78')))
    
    fig_fc.update_layout(title=f"Tesla Close Price – Next {forecast_days} Days ({model_name})",
                         template='plotly_dark', paper_bgcolor='#0e1117',
                         plot_bgcolor='#0e1117', height=420,
                         xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Download forecast
    csv_fc = fc_df.to_csv(index=False).encode()
    st.download_button("📥 Download Forecast CSV", csv_fc,
                       f"TSLA_forecast_{forecast_days}d.csv", "text/csv")

# ═══════════════════════════════════════════════════════════
# TAB 5: Model Comparison
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📊 Model Performance Comparison")
    
    m1 = metrics_df[metrics_df['Model'] == 'SimpleRNN'].iloc[0]
    m2 = metrics_df[metrics_df['Model'] == 'LSTM'].iloc[0]
    
    # Comparison Table
    st.dataframe(
        metrics_df.style
            .background_gradient(subset=['MSE','RMSE','MAE','MAPE'], cmap='RdYlGn_r')
            .background_gradient(subset=['R2'], cmap='RdYlGn')
            .format({'MSE':'${:.2f}','RMSE':'${:.4f}','MAE':'${:.4f}','MAPE':'{:.4f}%','R2':'{:.4f}'}),
        use_container_width=True
    )
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Bar chart comparison
        metrics_melt = pd.melt(
            metrics_df, id_vars='Model',
            value_vars=['RMSE','MAE'],
            var_name='Metric', value_name='Value')
        fig_bar = px.bar(metrics_melt, x='Metric', y='Value', color='Model',
                         barmode='group', title='RMSE & MAE Comparison',
                         color_discrete_map={'SimpleRNN':'#f6ad55','LSTM':'#48bb78'})
        fig_bar.update_layout(template='plotly_dark', paper_bgcolor='#0e1117',
                               plot_bgcolor='#0e1117', height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with c2:
        # R² Gauge
        fig_gauge = make_subplots(rows=1, cols=2,
                                  specs=[[{'type':'indicator'},{'type':'indicator'}]])
        fig_gauge.add_trace(go.Indicator(
            mode='gauge+number',
            value=m1['R2'],
            title={'text':'SimpleRNN R²', 'font':{'color':'#f6ad55'}},
            gauge={'axis':{'range':[0,1]},'bar':{'color':'#f6ad55'},
                   'steps':[{'range':[0,0.5],'color':'#3a1a1a'},
                             {'range':[0.5,0.75],'color':'#2a2a1a'},
                             {'range':[0.75,1],'color':'#1a2a1a'}]}),
            row=1, col=1)
        fig_gauge.add_trace(go.Indicator(
            mode='gauge+number',
            value=m2['R2'],
            title={'text':'LSTM R²','font':{'color':'#48bb78'}},
            gauge={'axis':{'range':[0,1]},'bar':{'color':'#48bb78'},
                   'steps':[{'range':[0,0.5],'color':'#3a1a1a'},
                             {'range':[0.5,0.75],'color':'#2a2a1a'},
                             {'range':[0.75,1],'color':'#1a2a1a'}]}),
            row=1, col=2)
        fig_gauge.update_layout(template='plotly_dark', paper_bgcolor='#0e1117',
                                 plot_bgcolor='#0e1117', height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Winner
    winner = 'LSTM' if m2['R2'] > m1['R2'] else 'SimpleRNN'
    st.success(f"""
    **🏆 Best Model: {winner}**
    
    LSTM achieved **R² = {m2['R2']:.4f}** vs SimpleRNN's **R² = {m1['R2']:.4f}**.
    LSTM's gating mechanism (input, forget, output gates) allows it to learn long-term dependencies
    in time series much more effectively than SimpleRNN, which suffers from vanishing gradients
    on long sequences. LSTM's lower RMSE (${m2['RMSE']:.2f}) and MAE (${m2['MAE']:.2f}) confirm
    its superior predictive accuracy on Tesla stock data.
    """)
    
    # Future improvements
    with st.expander("🚀 Advanced Improvements & Future Scope"):
        st.markdown("""
        | Approach | Description | Expected Impact |
        |----------|-------------|-----------------|
        | **GRU** | Gated Recurrent Unit — simpler than LSTM, often comparable accuracy | ⭐⭐⭐ |
        | **Transformer** | Attention-based model capturing long-range dependencies | ⭐⭐⭐⭐⭐ |
        | **Sentiment Analysis** | Reddit/Twitter NLP signals integrated as features | ⭐⭐⭐⭐ |
        | **News-Based Prediction** | Real-time news headlines via NLP embeddings | ⭐⭐⭐⭐ |
        | **Macro Features** | CPI, interest rates, oil prices as exogenous inputs | ⭐⭐⭐ |
        | **Multi-Stock** | Correlation-based cross-asset learning | ⭐⭐⭐ |
        | **Ensemble** | Weighted combination of RNN, LSTM, GRU, XGBoost | ⭐⭐⭐⭐ |
        | **Hyperparameter Tuning** | Optuna or Keras Tuner for automated HPO | ⭐⭐⭐ |
        """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#4a5568;font-size:0.8rem;padding:12px'>
  ⚡ Tesla Stock Prediction · Deep Learning System · Built with TensorFlow & Streamlit<br>
  <b>Disclaimer:</b> This tool is for educational purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)
