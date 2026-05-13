import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os

# Configuration
st.set_page_config(page_title="PhonePe Pulse Pro", layout="wide")

# Database Connection with Path Handling
@st.cache_resource
def get_connection():
    # Check multiple possible locations for the DB
    paths = ['phonepe_data.db', '../phonepe_data.db', 'streamlit_app/phonepe_data.db']
    for p in paths:
        if os.path.exists(p):
            return sqlite3.connect(p, check_same_thread=False)
    # If not found, just return the default (it will be empty)
    return sqlite3.connect('phonepe_data.db', check_same_thread=False)

conn = get_connection()

@st.cache_data
def load_data(table):
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        # Force column types to prevent comparison errors
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(int)
        return df
    except Exception:
        return pd.DataFrame()

# Load all required tables
df_agg_trans = load_data("aggregated_transaction")
df_map_trans = load_data("map_transaction")
df_agg_ins = load_data("aggregated_insurance")
df_top_trans = load_data("top_transaction")

# CRITICAL CHECK: If main table is empty, stop and warn user
if df_agg_trans.empty:
    st.error("🚨 **Database Error:** No data found in 'aggregated_transaction'.")
    st.info("Please ensure you ran `python scripts/etl_process.py` and that it successfully loaded rows.")
    st.stop()

# Sidebar Logic
st.sidebar.title("🔍 Filters")

# Safely get years
all_years = sorted(df_agg_trans['Year'].unique()) if 'Year' in df_agg_trans.columns else [2023]
selected_year = st.sidebar.selectbox("Select Year", all_years, index=len(all_years)-1)

# Safely get states
all_states = sorted(df_agg_trans['State'].unique()) if 'State' in df_agg_trans.columns else []
selected_state = st.sidebar.selectbox("Select State", ["All India"] + all_states)

# --- Helper function for safe filtering ---
def safe_filter(df, year, state):
    if df.empty:
        return df
    temp_df = df.copy()
    if 'Year' in temp_df.columns:
        temp_df = temp_df[temp_df['Year'] == year]
    if state != "All India" and 'State' in temp_df.columns:
        temp_df = temp_df[temp_df['State'] == state]
    return temp_df

# Filter all dataframes safely
filtered_trans = safe_filter(df_agg_trans, selected_year, selected_state)
filtered_map = safe_filter(df_map_trans, selected_year, selected_state)
filtered_ins = safe_filter(df_agg_ins, selected_year, selected_state)
filtered_top = safe_filter(df_top_trans, selected_year, selected_state)

# Header
st.title("📱 PhonePe Pulse Analytics")
st.caption(f"Showing insights for {selected_state} ({selected_year})")

# KPI Cards
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    val = filtered_trans['Amount'].sum() if not filtered_trans.empty else 0
    st.metric("Transaction Value", f"₹{val/1e7:,.2f} Cr")
with kpi2:
    cnt = filtered_trans['Count'].sum() if not filtered_trans.empty else 0
    st.metric("Transaction Count", f"{cnt/1e5:,.2f} Lakhs")
with kpi3:
    ins = filtered_ins['Amount'].sum() if not filtered_ins.empty else 0
    st.metric("Insurance Value", f"₹{ins/1e7:,.2f} Cr")

# Tabs
tabs = st.tabs(["Transactions", "Geography", "Insurance", "Insights"])

with tabs[0]:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Transaction Type Distribution")
        if not filtered_trans.empty and 'Type' in filtered_trans.columns:
            fig = px.pie(filtered_trans, values='Amount', names='Type', hole=0.5, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data for this selection.")
    with col_b:
        st.subheader("Yearly Growth Trend")
        if not df_agg_trans.empty:
            trend = df_agg_trans.groupby('Year')['Amount'].sum().reset_index()
            fig = px.line(trend, x='Year', y='Amount', markers=True, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Top Districts")
        if not filtered_map.empty and 'District' in filtered_map.columns:
            dist_data = filtered_map.groupby('District')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(dist_data, x='Amount', y='District', orientation='h', template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("District data (map_transaction) is missing.")
    with col_d:
        st.subheader("Top Pincodes")
        if not filtered_top.empty and 'Pincode' in filtered_top.columns:
            pin_data = filtered_top.groupby('Pincode')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(pin_data, x='Pincode', y='Amount', template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pincode data (top_transaction) is missing.")

with tabs[2]:
    st.subheader("Insurance Adoption Heatmap")
    if not df_agg_ins.empty:
        ins_h = df_agg_ins.groupby(['State', 'Year'])['Amount'].sum().reset_index()
        fig = px.density_heatmap(ins_h, x="Year", y="State", z="Amount", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insurance data not available.")

with tabs[3]:
    st.markdown("### 💡 Business Insights")
    st.write("- P2P payments lead in volume across all states.")
    st.write("- Tier 1 cities show higher insurance adoption rates.")