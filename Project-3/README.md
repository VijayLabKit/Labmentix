# 🚀 DeepCSAT — E-commerce Customer Support Analytics

An intelligent, production-ready data engineering and predictive modeling pipeline built to classify customer service interaction quality, forecast Customer Satisfaction (CSAT) scores, and uncover operational support-ticket bottlenecks.

---

## 📌 Problem Statement
Customer support latency, resolution quality, and workload distribution directly govern customer retention in e-commerce ecosystems. Suboptimal shift timings, processing bottlenecks, or unmitigated queues lead to a plunge in customer satisfaction.

**DeepCSAT** resolves this business dilemma by running an automated end-to-end data pipeline—cleaning raw ticket attributes, engineering operational features, and training robust machine learning models to map customer interaction vectors directly into real-time risk tiers. This enables managers to proactively flag and intercept friction points before a customer leaves a negative review.

---

## 📂 Project Structure

```text
Project-3/
│
├── data/
│   ├── raw/
│   │   └── eCommerce_Customer_support_data.csv # Raw baseline dataset
│   └── processed/
│       ├── cleaned_data.csv                    # Filtered intermediate records
│       └── featured_data.csv                   # Engineered analytical dataset
│
├── models/
│   └── csat_model.pkl                          # Serialized Model + Label Encoders
│
├── src/
│   ├── cleaning.py                             # Missing values and type parsing
│   ├── feature_engineering.py                  # Core metric calculations
│   └── model.py                                # Training, validation, and evaluation
│
├── app.py                                      # Real-time Streamlit web app
├── requirements.txt                            # Library dependencies
└── README.md                                   # Project documentation
```

---

## ⚙️ Modular Data Pipeline

The project features a decoupled data architecture separated into sequential stages:

### 1️⃣ Data Cleaning (`src/cleaning.py`)
Loads raw comma-separated files and enforces automated defensive sanitization routines to maximize downstream data quality:
*   **Imputation Vectors:** Programmatically isolates sparse categorical columns (`Customer Remarks`, `Customer_City`, `Product_category`, `Sub-category`) and treats missing fields by mapping them to `'Unknown'`.
*   **Numerical Alignment:** Coerces `Item_price` arrays safely into numerical representations while automatically bypassing syntax discrepancies using a median fallback value.
*   **Temporal Formatting:** Forces parsing over tracking timestamps (`Issue_reported at`, `issue_responded`, `Survey_response_Date`) to standardize datetime signatures.
*   **Target Enforcement:** Targets the critical `CSAT Score` feature, dropping any index row containing a null value and casting the valid elements into integers.
*   **Deduplication:** Scrubs duplicate record matrices to guarantee statistical independence across files, flushing the result to `cleaned_data.csv`.

### 2️⃣ Feature Engineering (`src/feature_engineering.py`)
Enforces high-utility feature computation while safely shielding internal code paths from `DtypeWarnings` and `Bin Edge` failures:
*   **Response Time Vector:** Programmatically processes the duration between `Issue_reported at` and `issue_responded` timestamps to yield `Response_Time_Min`, utilizing median replacement fallbacks for missing or negative intervals.
*   **Agent Queue Workload:** Automatically maps absolute frequency vectors across unique agent nodes using a mapping index dictionary to track `Agent_Total_Tickets`.
*   **Robust Cost Binning:** Computes an adaptive quantile partition structure over `Item_price` arrays using `pd.qcut` with built-in collision safety logic (`duplicates='drop'`) to seamlessly classify items into `Budget`, `Mid-range`, or `Premium` tiers.
*   **Temporal Hour Extractor:** Extracts the `Report_Hour` timestamp hour integer vector to map peak volume trends over a 24-hour cycle.

### 3️⃣ Model Training & Evaluation (`src/model.py`)
Isolates relevant variables and executes supervised ensemble training pipelines to export portable prediction engines:
*   **Target Vectors:** Extracts exactly 7 engineered core operational training dimensions: `channel_name`, `category`, `Tenure Bucket`, `Agent Shift`, `Response_Time_Min`, `Agent_Total_Tickets`, and `Report_Hour`.
*   **Label Encoding:** Isolates non-numeric object matrices dynamically and transforms text properties into mathematical label pointers via explicit Scikit-Learn `LabelEncoder` objects.
*   **Ensemble Training:** Generates an 80/20 train-test partition baseline splitting your records dynamically with a stable seed (`random_state=42`), feeding vectors into a `RandomForestClassifier` initialized with 100 decision estimators.
*   **Binary Payload Export:** Packages the trained tree framework alongside the compiled categorical encoders maps dictionary and target string feature headers into a standalone dictionary object payload, serializing it directly to `models/csat_model.pkl` via `joblib`.

---

## 🖥️ Streamlit App Features

The interactive enterprise application exposes a multi-tab workspace architecture:
*   **Project Overview:** Tracks foundational high-level performance indexes including *Average CSAT*, *Total System Ticket Intake*, and *Mean Response Latency (Minutes)*.
*   **Dataset Insights:** Renders interactive historical histograms, channel density charts, and response-time box plots.
*   **CSAT Prediction Tool:** Direct user input form mapping data fields directly into the serialized inference asset, displaying dynamic status indicators:
    *   🟢 **Satisfied** (Predicted Score $\ge$ 4)
    *   🟡 **Neutral** (Predicted Score = 3)
    *   🔴 **At Risk** (Predicted Score $\le$ 2)
*   **Model Insights:** Exposes model interpretability by graphing relative feature importance weights directly across user dashboards.

---

## 🚀 Installation & Usage Guide

### 1. Initialize Workspace Environment
```bash
cd E:\Github_Projects\Labmentix\Project-3
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Execute Data & Training Pipeline Sequences
Run the modular stages sequentially to reproduce results and serialize the prediction models from scratch:
```bash
# Step 1: Clean raw inputs and handle missing values
python src/cleaning.py

# Step 2: Compute engineered operational features
python src/feature_engineering.py

# Step 3: Run model training and validation loops
python src/model.py
```

### 3. Start the Local Production Server
```bash
streamlit run app.py
```

---

## 🛠️ Technology Stack
*   **Language & Runtime:** Python 3.10+
*   **Data Architecture:** Pandas, NumPy
*   **Interactive Visualizations:** Plotly Express Graphical Suite
*   **Serialization Management:** Joblib Binary Ingestion
*   **Machine Learning Backend:** Scikit-Learn Ensemble Estimators (`RandomForestClassifier`), `LabelEncoder`, and Validation Core metrics (`accuracy_score`, `classification_report`)
*   **User Interface Portal:** Streamlit Framework Architecture

---

## 👤 Author
*   **Ishan Chowdhury**  
*   Data Science with AI & Machine Learning Intern  
*   **Labmentix Internship Portfolio**
