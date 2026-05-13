import pandas as pd
import numpy as np
import os

def engineer_features(input_path, output_path):
    """
    Creates meaningful features for CSAT prediction.
    Addresses DtypeWarnings and Bin Edge errors.
    """
    # Use low_memory=False to avoid DtypeWarning for mixed types
    df = pd.read_csv(input_path, low_memory=False)
    
    # Convert dates safely
    date_cols = ['Issue_reported at', 'issue_responded']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 1. Response Time (in minutes)
    # Drop rows where dates are missing for this calculation to avoid errors
    mask = df['issue_responded'].notna() & df['Issue_reported at'].notna()
    df.loc[mask, 'Response_Time_Min'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60
    
    # Fill missing/negative response times with median
    median_response = df['Response_Time_Min'].median() if not df['Response_Time_Min'].empty else 0
    df['Response_Time_Min'] = df['Response_Time_Min'].fillna(median_response)
    df.loc[df['Response_Time_Min'] < 0, 'Response_Time_Min'] = median_response

    # 2. Agent Workload
    if 'Agent_name' in df.columns:
        agent_counts = df['Agent_name'].value_counts().to_dict()
        df['Agent_Total_Tickets'] = df['Agent_name'].map(agent_counts)
    else:
        df['Agent_Total_Tickets'] = 0

    # 3. Item Price Binning (FIXED for Duplicate Bin Edges)
    if 'Item_price' in df.columns:
        # Ensure Item_price is numeric
        df['Item_price'] = pd.to_numeric(df['Item_price'], errors='coerce').fillna(0)
        
        try:
            # Attempt quantile cut, dropping duplicates if many items have the same price
            # We use labels=False first to handle the potential drop in number of bins
            bins = pd.qcut(df['Item_price'], q=3, duplicates='drop')
            num_unique_bins = len(bins.unique())
            
            if num_unique_bins == 3:
                df['Price_Category'] = pd.qcut(df['Item_price'], q=3, labels=['Budget', 'Mid-range', 'Premium'], duplicates='drop')
            else:
                # Fallback if qcut collapses to fewer than 3 categories
                df['Price_Category'] = pd.cut(df['Item_price'], bins=3, labels=['Budget', 'Mid-range', 'Premium'])
        except Exception:
            # Final fallback if data is extremely skewed
            df['Price_Category'] = 'Standard'

    # 4. Hour of Report
    df['Report_Hour'] = df['Issue_reported at'].dt.hour.fillna(12).astype(int)
    
    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Featured dataset saved to {output_path}")
    return df

if __name__ == "__main__":
    # Ensure the paths match your structure
    input_file = 'data/processed/cleaned_data.csv'
    output_file = 'data/processed/featured_data.csv'
    
    if os.path.exists(input_file):
        engineer_features(input_file, output_file)
    else:
        print(f"Error: {input_file} not found. Did you run cleaning.py first?")