import pandas as pd
import os

def safe_merge(left_df, right_df, left_key, right_key, how='left'):
    """
    Safely merges two dataframes by aligning data types to strings,
    stripping whitespace, and handling missing columns or dataframes.
    """
    # 1. Validation: Check if dataframes are valid
    if left_df is None or left_df.empty:
        return left_df
    if right_df is None or right_df.empty:
        print(f"Warning: Skipping merge. Right dataframe for key '{right_key}' is empty or missing.")
        return left_df
    
    # 2. Check if keys exist in respective dataframes
    if left_key not in left_df.columns:
        print(f"Warning: Skipping merge. Left key '{left_key}' not found.")
        return left_df
    if right_key not in right_df.columns:
        print(f"Warning: Skipping merge. Right key '{right_key}' not found.")
        return left_df

    # 3. Data Type Alignment & Cleaning
    # Convert to string and strip whitespace to prevent "int64 vs object" errors
    left_df[left_key] = left_df[left_key].astype(str).str.strip().str.lstrip('0')
    right_df[right_key] = right_df[right_key].astype(str).str.strip().str.lstrip('0')

    # 4. Execute Merge
    try:
        merged_df = left_df.merge(right_df, left_on=left_key, right_on=right_key, how=how, suffixes=('', '_drop'))
        
        # 5. Post-Merge cleanup: Remove redundant duplicate columns
        cols_to_keep = [c for c in merged_df.columns if not c.endswith('_drop')]
        merged_df = merged_df[cols_to_keep]
        
        # Remove right_key if it's different from left_key and still exists
        if left_key != right_key and right_key in merged_df.columns:
            merged_df = merged_df.drop(columns=[right_key])
            
        return merged_df
    except Exception as e:
        print(f"Error merging '{left_key}' with '{right_key}': {e}")
        return left_df

def clean_data(raw_path='data/raw/', output_path='data/cleaned/'):
    """
    Relational data pipeline for Tourism Recommendation System.
    """
    os.makedirs(output_path, exist_ok=True)
    
    def load_csv(name):
        try:
            path = os.path.join(raw_path, f"{name}.csv")
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    # Load all tables
    df_trans = load_csv('Transaction')
    df_user = load_csv('User')
    df_item = load_csv('Updated_Item')
    df_city = load_csv('City')
    df_country = load_csv('Country')
    df_region = load_csv('Region')
    df_continent = load_csv('Continent')
    df_mode = load_csv('Mode')
    df_type = load_csv('Type')

    # Essential check
    if df_trans.empty:
        print("Error: Transaction dataset is missing or empty.")
        return

    # --- REPLACED WITH SAFE_MERGE ---
    
    # 1. Transaction + User
    master = safe_merge(df_trans, df_user, 'UserId', 'UserId')
    
    # 2. Transaction + Visit Mode (Fixes the int64 vs object error)
    master = safe_merge(master, df_mode, 'VisitMode', 'VisitModeId')
    
    # 3. Transaction + Attraction Details
    master = safe_merge(master, df_item, 'AttractionId', 'AttractionId')
    
    # 4. Master + Attraction Category
    master = safe_merge(master, df_type, 'AttractionTypeId', 'AttractionTypeId')
    
    # 5. Master + City Name
    master = safe_merge(master, df_city, 'AttractionCityId', 'CityId')
    
    # 6. Master + Country Name
    master = safe_merge(master, df_country, 'CountryId', 'CountryId')
    
    # 7. Master + Region Name
    master = safe_merge(master, df_region, 'RegionId', 'RegionId')
    
    # 8. Master + Continent Name
    master = safe_merge(master, df_continent, 'ContinentId', 'ContinentId')

    # --- DATA REFINEMENT ---

    # Numeric formatting
    if 'Rating' in master.columns:
        master['Rating'] = pd.to_numeric(master['Rating'], errors='coerce').fillna(3)
        master['HighRating'] = (master['Rating'] >= 4).astype(int)

    # Date components
    for col in ['VisitYear', 'VisitMonth']:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors='coerce').fillna(0).astype(int)

    # Text filling
    text_fields = ['CityName', 'Country', 'VisitMode', 'AttractionType', 'Attraction', 'Region', 'Continent']
    for col in text_fields:
        if col in master.columns:
            master[col] = master[col].fillna('Unknown')

    # Deduplication
    master = master.drop_duplicates()

    # Feature Engineering
    if 'CityName' in master.columns:
        master['CityPopularity'] = master.groupby('CityName')['TransactionId'].transform('count')
    if 'UserId' in master.columns:
        master['UserActivity'] = master.groupby('UserId')['TransactionId'].transform('count')

    # Save logic
    output_file = os.path.join(output_path, 'cleaned_data.csv')
    master.to_csv(output_file, index=False)

    print("-" * 40)
    print("SUCCESS: DATA CLEANING PIPELINE FINISHED")
    print(f"Rows count: {len(master)}")
    print(f"Columns count: {len(master.columns)}")
    print(f"File saved location: {output_file}")
    print("-" * 40)
    
    return master

if __name__ == "__main__":
    clean_data()