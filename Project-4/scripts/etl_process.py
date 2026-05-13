import os
import json
import pandas as pd
import sqlite3

# IMPORTANT: Replace with your local dataset path
DATA_PATH = "data" 

def extract_data():
    conn = sqlite3.connect('phonepe_data.db')
    
    categories = ['aggregated', 'map', 'top']
    sub_cats = ['transaction', 'user', 'insurance']
    
    for cat in categories:
        for sub in sub_cats:
            path = f"{DATA_PATH}/{cat}/{sub}/country/india/state/"
            if not os.path.exists(path):
                continue
                
            all_data = []
            states = os.listdir(path)
            
            for state in states:
                state_path = os.path.join(path, state)
                years = os.listdir(state_path)
                for year in years:
                    year_path = os.path.join(state_path, year)
                    files = os.listdir(year_path)
                    for file in files:
                        file_path = os.path.join(year_path, file)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                        quarter = int(file.strip('.json'))
                        state_name = state.replace("-", " ").title()
                        
                        # Logic branching based on Category and Sub-category
                        try:
                            if cat == 'aggregated' and sub == 'transaction':
                                for i in data['data']['transactionData']:
                                    all_data.append({
                                        "State": state_name, "Year": int(year), "Quarter": quarter,
                                        "Type": i['name'], "Count": i['paymentInstruments'][0]['count'],
                                        "Amount": i['paymentInstruments'][0]['amount']
                                    })
                            
                            elif cat == 'aggregated' and sub == 'user':
                                users = data['data'].get('usersByDevice')
                                if users:
                                    for i in users:
                                        all_data.append({
                                            "State": state_name, "Year": int(year), "Quarter": quarter,
                                            "Brand": i['brand'], "Count": i['count']
                                        })
                            
                            elif cat == 'map' and sub == 'transaction':
                                for i in data['data']['hoverDataList']:
                                    all_data.append({
                                        "State": state_name, "Year": int(year), "Quarter": quarter,
                                        "District": i['name'].title(), "Count": i['metric'][0]['count'],
                                        "Amount": i['metric'][0]['amount']
                                    })
                                    
                            elif cat == 'top' and sub == 'transaction':
                                for i in data['data']['pincodes']:
                                    all_data.append({
                                        "State": state_name, "Year": int(year), "Quarter": quarter,
                                        "Pincode": i['entityName'], "Count": i['metric']['count'],
                                        "Amount": i['metric']['amount']
                                    })
                            # Add Insurance processing
                            elif sub == 'insurance':
                                insurance_data = data['data'].get('insuranceData', data['data'].get('hoverDataList', []))
                                for i in insurance_data:
                                    all_data.append({
                                        "State": state_name, "Year": int(year), "Quarter": quarter,
                                        "Type": i.get('name', 'General'), "Count": i.get('count', i.get('metric', [{}])[0].get('count')),
                                        "Amount": i.get('amount', i.get('metric', [{}])[0].get('amount'))
                                    })
                        except (KeyError, TypeError) as e:
                            print(f"Skipping malformed data in {file_path}: {e}")
            
            if all_data:
                table_name = f"{cat}_{sub}"
                df = pd.DataFrame(all_data)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Loaded {len(df)} rows into {table_name}")

    conn.close()

if __name__ == "__main__":
    print("🚀 Starting Production ETL...")
    extract_data()
    print("✅ ETL Complete. Database 'phonepe_data.db' is ready.")