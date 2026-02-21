import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tourism Insights & Recommendations", layout="wide")

# --- DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    path = 'data/cleaned/cleaned_data.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_ml():
    try:
        # Using standard open/pickle for compatibility
        reg = pickle.load(open('models/reg_model.pkl', 'rb'))
        clf = pickle.load(open('models/clf_model.pkl', 'rb'))
        enc = pickle.load(open('models/encoders.pkl', 'rb'))
        feats = pickle.load(open('models/feature_names.pkl', 'rb'))
        return reg, clf, enc, feats
    except:
        return None, None, None, None

df = load_data()
reg, clf, encoders, features = load_ml()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Travel Explorer")
page = st.sidebar.radio("Navigate", ["Dashboard", "Analytics", "Predictor", "Recommendations"])

# --- UI LOGIC ---
if df is None:
    st.error("Cleaned data not found. Please run 'python src/cleaning.py' first.")
else:
    if page == "Dashboard":
        st.title("🏠 Tourism Dashboard")
        
        # KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Trips", f"{len(df):,}")
        kpi2.metric("Avg Rating", f"{df['Rating'].mean():.2f} ⭐")
        kpi3.metric("Unique Cities", df['CityName'].nunique())
        
        top_country = "N/A"
        if 'Country' in df.columns and not df['Country'].empty:
            top_country = df['Country'].mode()[0]
        kpi4.metric("Top Country", top_country)
        
        # Charts Section
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Top 10 Cities by Visits")
            if 'CityName' in df.columns:
                top_cities = df['CityName'].value_counts().head(10).reset_index()
                top_cities.columns = ['CityName', 'count']
                
                fig = px.bar(
                    top_cities, 
                    x='CityName', 
                    y='count', 
                    color='CityName',
                    labels={'CityName': 'City', 'count': 'Number of Visits'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Column 'CityName' missing for visualization.")
            
        with c2:
            st.subheader("Travel Mode Distribution")
            if 'VisitMode' in df.columns:
                mode_dist = df['VisitMode'].value_counts().reset_index()
                mode_dist.columns = ['VisitMode', 'count']
                
                fig2 = px.pie(
                    mode_dist, 
                    values='count', 
                    names='VisitMode', 
                    hole=0.4
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Column 'VisitMode' missing for visualization.")

    elif page == "Analytics":
        st.title("📊 Deep Dive Analytics")
        
        countries = ["All"] + sorted(list(df['Country'].unique())) if 'Country' in df.columns else ["All"]
        selected_country = st.selectbox("Select Country", countries)
        
        filtered_df = df if selected_country == "All" else df[df['Country'] == selected_country]
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader(f"Attraction Types in {selected_country}")
            if 'AttractionType' in filtered_df.columns:
                type_count = filtered_df['AttractionType'].value_counts().reset_index()
                type_count.columns = ['AttractionType', 'count']
                
                fig3 = px.funnel(type_count, x='count', y='AttractionType')
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Column 'AttractionType' missing.")

        with col_b:
            st.subheader("Average Rating over Years")
            if 'VisitYear' in filtered_df.columns and 'Rating' in filtered_df.columns:
                yearly_rating = filtered_df.groupby('VisitYear')['Rating'].mean().reset_index()
                yearly_rating.columns = ['VisitYear', 'AvgRating']
                
                fig4 = px.line(yearly_rating, x='VisitYear', y='AvgRating', markers=True)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("Date or Rating columns missing for trend analysis.")

    elif page == "Predictor":
        st.title("🤖 Rating Predictor")
        if reg is None or encoders is None:
            st.warning("Machine Learning models are not trained. Please run 'python src/models.py'.")
        else:
            with st.container():
                st.info("Input trip details to predict the expected rating and experience quality.")
                
                with st.form("predict_form"):
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        in_year = st.number_input("Year", 2023, 2030, 2024)
                        in_month = st.slider("Month", 1, 12, 6)
                        in_mode = st.selectbox("Travel Mode", df['VisitMode'].unique())
                    with col_p2:
                        in_type = st.selectbox("Attraction Type", df['AttractionType'].unique())
                        in_city = st.selectbox("City", df['CityName'].unique())
                    
                    submit = st.form_submit_button("Predict Experience Quality")
                    
                    if submit:
                        input_data = pd.DataFrame([[in_year, in_month, in_mode, in_type, in_city]], columns=features)
                        
                        for col, le in encoders.items():
                            if col in input_data.columns:
                                try:
                                    input_data[col] = le.transform(input_data[col].astype(str))
                                except:
                                    input_data[col] = 0 
                        
                        try:
                            p_rating = reg.predict(input_data)[0]
                            p_class = clf.predict(input_data)[0]
                            
                            st.write("---")
                            res_col1, res_col2 = st.columns(2)
                            res_col1.metric("Predicted Score", f"{p_rating:.2f} / 5.0")
                            
                            if p_class == 1:
                                res_col2.success("Verdict: Highly Recommended! 🌟")
                            else:
                                res_col2.info("Verdict: Standard Experience.")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

    elif page == "Recommendations":
        st.title("✈ Travel Recommendations")
        
        user_input = st.text_input("Enter User ID (e.g. 0000070456)").strip().lstrip('0')
        
        if user_input:
            user_trips = df[df['UserId'].astype(str).str.lstrip('0') == user_input]
            
            if not user_trips.empty:
                st.success(f"Found {len(user_trips)} previous trips for User {user_input}.")
                with st.expander("View Travel History"):
                    st.table(user_trips[['VisitYear', 'CityName', 'Attraction', 'Rating']].head(10))
                
                fav_types = user_trips[user_trips['Rating'] >= 4]['AttractionType'].unique()
                if len(fav_types) > 0:
                    st.subheader("Personalized Picks for You")
                    recs = df[df['AttractionType'].isin(fav_types) & (~df['Attraction'].isin(user_trips['Attraction']))]
                    recs = recs.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(5).reset_index()
                    recs.columns = ['Attraction', 'AvgRating']
                    
                    for _, row in recs.iterrows():
                        st.write(f"✅ **{row['Attraction']}** ({row['AvgRating']:.1f} ⭐)")
                else:
                    st.info("No high-rated history found. Showing popular attractions instead.")
                    pops = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(5).reset_index()
                    pops.columns = ['Attraction', 'AvgRating']
                    for _, row in pops.iterrows():
                        st.write(f"📍 **{row['Attraction']}** ({row['AvgRating']:.1f} ⭐)")
            else:
                st.warning(f"User ID {user_input} not found in database.")
                st.subheader("Global Popular Destinations")
                pop_cities = df['CityName'].value_counts().head(5).reset_index()
                pop_cities.columns = ['CityName', 'VisitCount']
                st.table(pop_cities)
        else:
            st.info("Enter a User ID for personalized results.")
            st.subheader("Most Visited Cities Right Now")
            top_visited = df['CityName'].value_counts().head(5).reset_index()
            top_visited.columns = ['CityName', 'Visits']
            st.bar_chart(top_visited.set_index('CityName'))