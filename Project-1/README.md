# 🍽️ Zomato Restaurant Review Analysis  
### Labmentix Internship Task

**Name:** Ishan Chowdhury  
**Domain:** Data Science with AI & Machine Learning  
**Internship:** Labmentix  

---

## 📌 Project Description

This project is developed as part of my **Labmentix Internship Task** under the **Data Science with AI & Machine Learning** domain.

The objective of this project is to analyze **Zomato restaurant reviews**, perform **sentiment analysis using NLP**, conduct **exploratory data analysis (EDA)**, and build a **machine learning model** to predict restaurant ratings based on customer sentiment and other influencing factors.

---

## 🎯 Project Objectives

- Analyze customer reviews using Natural Language Processing (NLP)
- Classify reviews into Positive, Neutral, and Negative sentiments
- Visualize trends between cost, rating, and sentiment
- Build a machine learning model to predict restaurant ratings

---

## 🔍 What the Project Does

### 1️⃣ Data Loading & Merging
- Loads Zomato reviews and metadata datasets
- Merges them into a single dataset

### 2️⃣ Data Cleaning & Preprocessing
- Converts cost to numeric
- Cleans ratings
- Extracts reviewer activity data
- Handles missing values

### 3️⃣ Sentiment Analysis
- Uses **VADER Sentiment Analyzer**
- Generates sentiment score and label

### 4️⃣ Exploratory Data Analysis
- Sentiment distribution
- Cost vs Rating analysis
- Top cuisines visualization

### 5️⃣ Machine Learning
- Random Forest Regressor
- Predicts restaurant ratings
- Evaluates using MSE and R² score

---

## 📂 Project Structure

```
Zomato_Analysis/
├── zomato_analysis.py
├── Zomato Restaurant reviews.csv
├── Zomato Restaurant names and Metadata.csv
├── Processed_Zomato_Data.csv
├── sentiment_distribution.png
├── cost_vs_rating.png
├── top_cuisines.png
├── feature_importance.png
└── README.md
```

---

## 📊 Output Generated

- Processed_Zomato_Data.csv
- Sentiment and EDA visualizations
- Feature importance plot

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- VADER Sentiment Analysis
- Scikit-learn

---

## ▶️ How to Run

### Step 1: Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn vaderSentiment
```

### Step 2: Run Script
```bash
python zomato_analysis.py
```

---

## 🏁 Conclusion

This project demonstrates an end-to-end data science workflow combining NLP, EDA, and machine learning, completed as part of the **Labmentix Internship**.

---

## 👤 Author

**Ishan Chowdhury**  
Data Science with AI & Machine Learning Intern  
**Labmentix**
