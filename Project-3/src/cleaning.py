import pandas as pd
import numpy as np
import os

def clean_data(input_path, output_path):
    """
    Loads raw dataset, handles missing values, and fixes data types.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # 1. Handling Missing Values
    # Categorical: Fill with 'Unknown'
    cat_cols = ['Customer Remarks', 'Customer_City', 'Product_category', 'Sub-category']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
    # Numerical: Fill with median
    if 'Item_price' in df.columns:
        df['Item_price'] = pd.to_numeric(df['Item_price'], errors='coerce')
        df['Item_price'] = df['Item_price'].fillna(df['Item_price'].median())
        
    # 2. Date Conversions
    date_cols = ['Issue_reported at', 'issue_responded', 'Survey_response_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. Handle Target Variable (CSAT Score)
    if 'CSAT Score' in df.columns:
        # Drop rows where target is missing
        df = df.dropna(subset=['CSAT Score'])
        df['CSAT Score'] = df['CSAT Score'].astype(int)

    # 4. Remove Duplicates
    df = df.drop_duplicates()

    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_data('data/raw/eCommerce_Customer_support_data.csv', 'data/processed/cleaned_data.csv')