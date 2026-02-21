Tourism Recommendation System

An end-to-end internship project that analyzes travel data, predicts user ratings, and provides personalized travel recommendations.

🚀 Features

Data Pipeline: Automated cleaning and joining of relational CSV datasets.

Interactive Dashboard: Real-time KPIs and geographic insights using Streamlit.

Machine Learning:

Random Forest Regressor for rating estimation.

Random Forest Classifier for quality labeling.

Recommendations: User-history based or popularity-based suggestions.

🛠 Technologies Used

Frontend: Streamlit

Data: Pandas, Plotly

ML: Scikit-Learn

Storage: Local CSV & Pickle

📂 Installation & Running

Install dependencies: pip install -r requirements.txt

Prepare data: Ensure CSVs are in data/raw/

Run Cleaning: python src/cleaning.py

Run Training: python src/models.py

Launch App: streamlit run app.py