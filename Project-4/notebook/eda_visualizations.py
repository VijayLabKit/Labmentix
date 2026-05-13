import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORTANT: Replace with your local dataset path
conn = sqlite3.connect("phonepe_data.db")


def run_visuals():
    # Load data from SQL
    df_trans = pd.read_sql("SELECT * FROM aggregated_transaction", conn)

    # 1. Bar Chart: Transaction Amount by Year
    plt.figure(figsize=(10, 6))
    yearly_data = df_trans.groupby("Year")["Transaction_amount"].sum().reset_index()
    sns.barplot(x="Year", y="Transaction_amount", data=yearly_data, palette="viridis")
    plt.title("Total Transaction Amount Year-over-Year")
    plt.ylabel("Amount (in Billions)")
    plt.savefig("reports/yearly_trend.png")

    # 2. Pie Chart: Transaction Type Share
    plt.figure(figsize=(8, 8))
    type_data = df_trans.groupby("Transaction_type")["Transaction_count"].sum()
    type_data.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Market Share by Transaction Type")
    plt.ylabel("")
    plt.savefig("reports/transaction_types.png")


if __name__ == "__main__":
    run_visuals()
    print("Visualizations saved to reports folder.")
