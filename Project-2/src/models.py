import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score

def train_models(data_path='data/cleaned/cleaned_data.csv', model_dir='models/'):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Selection of simple features
    features = ['VisitYear', 'VisitMonth', 'VisitMode', 'AttractionType', 'CityName']
    features = [f for f in features if f in df.columns]
    
    X = df[features].copy()
    y_reg = df['Rating']
    y_clf = df['HighRating']
    
    # Encoding
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # Split
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    
    # Train Regressor
    reg = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    reg.fit(X_train, y_reg_train)
    reg_preds = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, reg_preds)
    
    # Train Classifier
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(X_train, y_clf_train)
    clf_preds = clf.predict(X_test)
    acc = accuracy_score(y_clf_test, clf_preds)
    
    # Save everything
    with open(f'{model_dir}reg_model.pkl', 'wb') as f: pickle.dump(reg, f)
    with open(f'{model_dir}clf_model.pkl', 'wb') as f: pickle.dump(clf, f)
    with open(f'{model_dir}encoders.pkl', 'wb') as f: pickle.dump(encoders, f)
    with open(f'{model_dir}feature_names.pkl', 'wb') as f: pickle.dump(features, f)
    
    print(f"Models Trained. Regression MAE: {mae:.2f}, Classification Acc: {acc:.2f}")

if __name__ == "__main__":
    train_models()