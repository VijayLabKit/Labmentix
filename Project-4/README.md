# 📊 PhonePe Pulse Data Visualization Project

An **end-to-end data engineering and analytics project** that extracts,
transforms, and visualizes PhonePe transaction and user growth data
across India.

This project processes the **PhonePe Pulse dataset (2018--2023)** and
builds an **interactive Streamlit dashboard** to explore transaction
trends, regional growth, and insurance adoption.

------------------------------------------------------------------------

# 📌 Project Overview

The **PhonePe Pulse dataset** consists of thousands of JSON files
containing detailed information about:

-   Transaction data
-   User adoption statistics
-   Insurance metrics
-   Geographic payment distribution

This project:

1.  Extracts raw JSON data
2.  Transforms it into structured tables
3.  Stores it inside an **SQLite database**
4.  Visualizes insights using **Streamlit dashboards**

------------------------------------------------------------------------

# 📂 Project Folder Structure

    phonepe_project/
    │
    ├── data/                   # Place PhonePe Pulse JSON dataset here
    │   ├── aggregated/         # Aggregated transaction & user data
    │   ├── map/                # District level hover data
    │   └── top/                # Pincode level data
    │
    ├── scripts/
    │   └── etl_process.py      # ETL pipeline script
    │
    ├── streamlit_app/
    │   └── app.py              # Streamlit dashboard
    │
    ├── requirements.txt        # Project dependencies
    ├── phonepe_data.db         # SQLite database generated after ETL
    └── README.md               # Project documentation

------------------------------------------------------------------------

# 🛠️ Prerequisites

Make sure the following tools are installed:

-   **Python 3.8+**
-   **VS Code** (recommended)
-   **PhonePe Pulse Dataset**

Download the dataset and place it inside the **data/** directory.

------------------------------------------------------------------------

# 🚀 Getting Started

## 1️⃣ Setup the Project

Clone or download this project and ensure the folder structure matches
the one shown above.

Place the following folders inside `data/`:

-   `aggregated`
-   `map`
-   `top`

------------------------------------------------------------------------

## 2️⃣ Install Dependencies

Open the terminal in the project folder and run:

    pip install -r requirements.txt

This installs:

-   pandas
-   sqlite3
-   streamlit
-   plotly
-   and other required libraries.

------------------------------------------------------------------------

## 3️⃣ Run the ETL Pipeline

The ETL script will:

-   Read all JSON files
-   Clean and transform the data
-   Store the results inside **SQLite database**

Run:

    python scripts/etl_process.py

After successful execution, the terminal should show:

    ✅ Loaded aggregated transaction data
    ✅ Loaded aggregated user data
    ✅ Loaded map transaction data
    ✅ Loaded top transaction data

This creates the database file:

    phonepe_data.db

------------------------------------------------------------------------

# 📊 Launch the Dashboard

Start the **Streamlit dashboard**:

    streamlit run streamlit_app/app.py

Streamlit will automatically open the dashboard in your browser.

Default URL:

    http://localhost:8501

------------------------------------------------------------------------

# 📊 Dashboard Features

### 📌 Key Metrics

-   Total Transaction Value
-   Total Transaction Count
-   Insurance Adoption

### 🌍 Geographic Insights

-   Top 10 districts by transaction value
-   Top pincodes by payment volume

### 📈 Growth Analysis

-   Year-wise transaction trends
-   Quarter-wise growth visualization

### 🎛 Interactive Filters

Users can filter the dashboard by:

-   Year
-   State (e.g., Maharashtra, Karnataka, Tamil Nadu)

------------------------------------------------------------------------

# 💡 Key Insights Generated

### 📊 Transaction Patterns

Festive quarters (**Q4**) show peak transaction volumes.

### 🌍 Regional Growth

States can be categorized into:

-   **High-growth markets**
-   **Mature digital payment regions**

### 🛡 Insurance Trends

Visualization of digital insurance adoption across Indian states.

------------------------------------------------------------------------

# 🛠 Troubleshooting

### ❌ KeyError: 'Year'

Cause: Database tables are empty.

Fix: Run the ETL pipeline again:

    python scripts/etl_process.py

Also confirm that the `data/` folder contains JSON files.

------------------------------------------------------------------------

### ❌ Port Already in Use

Run Streamlit on another port:

    streamlit run streamlit_app/app.py --server.port 8502

------------------------------------------------------------------------

# 📚 Tech Stack

-   **Python**
-   **Pandas**
-   **SQLite**
-   **Streamlit**
-   **Plotly**
-   **JSON Processing**

------------------------------------------------------------------------

# 👨‍💻 Author

Data Engineering & Analytics Project based on **PhonePe Pulse Dataset**.

------------------------------------------------------------------------

# ⭐ If you like this project

Consider giving the repository a **star ⭐ on GitHub**.
