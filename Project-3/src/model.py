import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os

def train_model(data_path, model_save_path):
    df = pd.read_csv(data_path)
    
    # Selecting features
    features = ['channel_name', 'category', 'Tenure Bucket', 'Agent Shift', 
                'Response_Time_Min', 'Agent_Total_Tickets', 'Report_Hour']
    
    # Dropping rows with missing values in selected features
    df = df.dropna(subset=features + ['CSAT Score'])
    
    X = df[features].copy()
    y = df['CSAT Score']

    # Label Encoding for categorical strings
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model: Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and encoders
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    payload = {
        'model': model,
        'encoders': encoders,
        'feature_names': features
    }
    joblib.dump(payload, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model('data/processed/featured_data.csv', 'models/csat_model.pkl')