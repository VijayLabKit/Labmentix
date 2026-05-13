import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# Set page config
st.set_page_config(page_title="DeepCSAT Dashboard", layout="wide")

# App setup
st.title("🚀 DeepCSAT - E-commerce Customer Support Analytics")
st.markdown("Predicting and Analyzing Customer Satisfaction Scores")

# Sidebar
menu = ["Project Overview", "Dataset Insights", "CSAT Prediction Tool", "Model Insights"]
choice = st.sidebar.selectbox("Navigation", menu)

# Load Data helper
@st.cache_data
def load_data():
    if os.path.exists('data/processed/featured_data.csv'):
        return pd.read_csv('data/processed/featured_data.csv')
    return pd.read_csv('eCommerce_Customer_support_data.csv')

df = load_data()

if choice == "Project Overview":
    st.header("Business Problem")
    st.write("""
    Customer support is the backbone of e-commerce. Low CSAT scores often indicate unresolved issues, 
    agent inefficiency, or product quality problems. This project predicts CSAT scores based on ticket 
    attributes to help managers intervene before a customer leaves a negative review.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg CSAT", round(df['CSAT Score'].mean(), 2) if 'CSAT Score' in df.columns else "N/A")
    col2.metric("Total Tickets", len(df))
    col3.metric("Avg Response (Min)", round(df['Response_Time_Min'].mean(), 2) if 'Response_Time_Min' in df.columns else "N/A")

elif choice == "Dataset Insights":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CSAT Distribution")
        if 'CSAT Score' in df.columns:
            fig = px.histogram(df, x='CSAT Score', color='CSAT Score', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.subheader("Tickets by Channel")
        if 'channel_name' in df.columns:
            fig = px.pie(df, names='channel_name', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Response Time vs CSAT Score")
    if 'Response_Time_Min' in df.columns:
        fig = px.box(df, x='CSAT Score', y='Response_Time_Min', color='CSAT Score')
        st.plotly_chart(fig, use_container_width=True)

elif choice == "CSAT Prediction Tool":
    st.header("Predict Customer Satisfaction")
    
    if os.path.exists('models/csat_model.pkl'):
        payload = joblib.load('models/csat_model.pkl')
        model = payload['model']
        encoders = payload['encoders']
        
        st.info("Enter interaction details below to predict the likely CSAT score.")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            channel = col1.selectbox("Channel", ["Inbound", "Outcall", "Email", "Chat"])
            category = col2.selectbox("Issue Category", df['category'].unique())
            tenure = col1.selectbox("Agent Tenure", [">90", "On Job Training", "31-60", "61-90", "0-30"])
            shift = col2.selectbox("Agent Shift", ["Morning", "Evening", "Night"])
            resp_time = st.slider("Estimated Response Time (Minutes)", 0, 500, 30)
            workload = st.number_input("Agent Tickets Today", 1, 100, 5)
            hour = st.slider("Hour of Day", 0, 23, 12)
            
            submit = st.form_submit_button("Predict CSAT")
            
            if submit:
                # Prepare input
                input_data = pd.DataFrame([{
                    'channel_name': channel,
                    'category': category,
                    'Tenure Bucket': tenure,
                    'Agent Shift': shift,
                    'Response_Time_Min': resp_time,
                    'Agent_Total_Tickets': workload,
                    'Report_Hour': hour
                }])
                
                # Encode
                for col, le in encoders.items():
                    # Handle unseen labels by defaulting to first class
                    input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
                
                prediction = model.predict(input_data)[0]
                
                if prediction >= 4:
                    st.success(f"Predicted CSAT Score: {prediction} (Satisfied)")
                elif prediction == 3:
                    st.warning(f"Predicted CSAT Score: {prediction} (Neutral)")
                else:
                    st.error(f"Predicted CSAT Score: {prediction} (At Risk)")
    else:
        st.error("Model file not found. Please train the model first.")

elif choice == "Model Insights":
    st.header("Feature Importance")
    if os.path.exists('models/csat_model.pkl'):
        payload = joblib.load('models/csat_model.pkl')
        model = payload['model']
        features = payload['feature_names']
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color='Importance')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Train the model to see feature importance.")