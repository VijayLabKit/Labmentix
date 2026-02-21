# 🌍 Tourism Recommendation System

### Labmentix Internship Task

**Name:** Ishan Chowdhury\
**Domain:** Data Science with AI & Machine Learning\
**Internship:** Labmentix

------------------------------------------------------------------------

## 📌 Project Description

This project was developed as part of my **Labmentix Internship Task**
under the **Data Science with AI & Machine Learning** domain.

The aim of this project is to analyze tourism and travel datasets, build
machine learning models to predict user ratings, and create a
personalized recommendation system that suggests destinations based on
user behavior and popularity trends.

------------------------------------------------------------------------

## 🎯 Project Objectives

-   Perform data cleaning and preprocessing on travel datasets\
-   Merge relational CSV files into a structured dataset\
-   Conduct exploratory data analysis (EDA)\
-   Build machine learning models for prediction and classification\
-   Develop a personalized tourism recommendation system\
-   Deploy an interactive dashboard for insights

------------------------------------------------------------------------

## 🔍 What the Project Does

### 1️⃣ Data Loading & Cleaning

-   Loads multiple tourism-related CSV datasets\
-   Cleans missing and inconsistent values\
-   Merges relational datasets into a unified structure

### 2️⃣ Exploratory Data Analysis

-   Identifies travel trends and rating distributions\
-   Generates key performance indicators (KPIs)\
-   Provides geographic and popularity insights

### 3️⃣ Machine Learning

-   **Random Forest Regressor** for predicting user ratings\
-   **Random Forest Classifier** for quality categorization\
-   Model evaluation using performance metrics

### 4️⃣ Recommendation System

-   Personalized recommendations based on user history\
-   Popularity-based fallback suggestions\
-   Intelligent filtering logic for better suggestions

### 5️⃣ Interactive Dashboard

-   Built using Streamlit\
-   Displays real-time KPIs\
-   Interactive charts and recommendation outputs

------------------------------------------------------------------------

## 📂 Project Structure

    Tourism_Recommendation_System/
    │
    ├── app.py
    ├── src/
    │   ├── cleaning.py
    │   ├── models.py
    │
    ├── data/
    │   └── raw/
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 📊 Output Generated

-   Cleaned dataset\
-   Trained machine learning models (.pkl files)\
-   Interactive Streamlit dashboard\
-   Personalized destination recommendations

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python\
-   Pandas\
-   Plotly\
-   Scikit-learn\
-   Streamlit\
-   Local CSV & Pickle storage

------------------------------------------------------------------------

## ▶️ How to Run

### Step 1: Install Dependencies

    pip install -r requirements.txt

### Step 2: Prepare Data

Place all raw CSV files inside:

    data/raw/

### Step 3: Run Data Cleaning

    python src/cleaning.py

### Step 4: Train Models

    python src/models.py

### Step 5: Launch Application

    streamlit run app.py

------------------------------------------------------------------------

## 🏁 Conclusion

This project demonstrates a complete end-to-end machine learning
pipeline, integrating data preprocessing, predictive modeling,
recommendation systems, and dashboard deployment. It highlights
practical implementation of ML concepts in a real-world tourism domain
as part of the **Labmentix Internship**.
