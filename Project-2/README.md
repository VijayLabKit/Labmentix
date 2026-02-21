🌍 Tourism Recommendation System
Labmentix Internship Task

Name: Ishan Chowdhury
Domain: Data Science with AI & Machine Learning
Internship: Labmentix

📌 Project Description

This project is developed as part of my Labmentix Internship Task under the Data Science with AI & Machine Learning domain.

The objective of this project is to analyze travel-related datasets, build predictive models for user ratings, and develop a personalized tourism recommendation system based on user behavior and popularity trends.

🎯 Project Objectives

Clean and merge multiple relational travel datasets

Perform exploratory data analysis (EDA)

Build machine learning models for rating prediction and classification

Develop a recommendation engine

Create an interactive dashboard for real-time insights

🔍 What the Project Does
1️⃣ Data Loading & Processing

Loads multiple travel-related CSV datasets

Cleans and merges relational data

Handles missing values and formatting issues

2️⃣ Exploratory Data Analysis

Generates key performance indicators (KPIs)

Identifies rating trends and travel patterns

Provides geographic insights

3️⃣ Machine Learning

Random Forest Regressor for user rating prediction

Random Forest Classifier for quality labeling

4️⃣ Recommendation System

User-history based personalized recommendations

Popularity-based fallback recommendations

5️⃣ Interactive Dashboard

Built using Streamlit

Displays real-time KPIs

Interactive charts and travel insights

📂 Project Structure
Tourism_Recommendation_System/
├── app.py
├── src/
│   ├── cleaning.py
│   ├── models.py
├── data/
│   └── raw/
├── requirements.txt
└── README.md
📊 Output Generated

Cleaned dataset

Trained ML models (Pickle files)

Interactive Streamlit dashboard

Personalized travel recommendations

🛠️ Technologies Used

Python

Pandas

Plotly

Scikit-learn

Streamlit

Local CSV & Pickle storage

▶️ How to Run
Step 1: Install Dependencies
pip install -r requirements.txt
Step 2: Prepare Data

Ensure CSV files are placed inside:

data/raw/
Step 3: Run Data Cleaning
python src/cleaning.py
Step 4: Train Models
python src/models.py
Step 5: Launch Application
streamlit run app.py
🏁 Conclusion

This project demonstrates an end-to-end machine learning system combining data preprocessing, predictive modeling, recommendation systems, and interactive dashboard development, completed as part of the Labmentix Internship.
