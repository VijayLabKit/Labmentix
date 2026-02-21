import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load and merge the Zomato datasets."""
    print("Loading datasets...")
    try:
        reviews_df = pd.read_csv('Zomato Restaurant reviews.csv')
        metadata_df = pd.read_csv('Zomato Restaurant names and Metadata.csv')

        df = pd.merge(
            reviews_df,
            metadata_df,
            left_on='Restaurant',
            right_on='Name',
            how='left'
        )

        print(f"Data merged successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean and preprocess the merged dataset."""
    print("Cleaning data...")

    df['Cost'] = df['Cost'].astype(str).str.replace(',', '').astype(float)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

    def extract_metadata(text):
        if pd.isna(text):
            return 0, 0
        reviews = re.findall(r'(\d+) Review', str(text))
        followers = re.findall(r'(\d+) Follower', str(text))
        return int(reviews[0]) if reviews else 0, int(followers[0]) if followers else 0

    df[['Reviewer_Reviews', 'Reviewer_Followers']] = df['Metadata'].apply(
        lambda x: pd.Series(extract_metadata(x))
    )

    df['Review'] = df['Review'].fillna('')
    df.dropna(subset=['Rating'], inplace=True)

    return df

def perform_sentiment_analysis(df):
    """Calculate sentiment score using VADER."""
    print("Performing Sentiment Analysis...")

    def get_sentiment(text):
        return analyzer.polarity_scores(str(text))['compound']

    df['Sentiment_Score'] = df['Review'].apply(get_sentiment)

    df['Sentiment_Label'] = df['Sentiment_Score'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )

    return df

def exploratory_analysis(df):
    """Generate visualizations for EDA."""
    print("Generating EDA plots...")

    plt.figure()
    sns.countplot(x='Sentiment_Label', data=df)
    plt.title('Distribution of Review Sentiments')
    plt.savefig('sentiment_distribution.png')

    plt.figure()
    sns.scatterplot(x='Cost', y='Rating', hue='Sentiment_Label', data=df)
    plt.title('Cost vs Rating (colored by Sentiment)')
    plt.savefig('cost_vs_rating.png')

    plt.figure()
    top_cuisines = df['Cuisines'].value_counts().head(10)
    sns.barplot(x=top_cuisines.values, y=top_cuisines.index)
    plt.title('Top 10 Most Common Cuisines')
    plt.savefig('top_cuisines.png')

def build_rating_predictor(df):
    """Build a model to predict Rating."""
    print("Building Prediction Model...")

    X = df[['Cost', 'Sentiment_Score', 'Reviewer_Reviews', 'Reviewer_Followers']]
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model Results -> MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    plt.figure()
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(4).plot(kind='barh')
    plt.title('Feature Importance for Rating Prediction')
    plt.savefig('feature_importance.png')

def main():
    df = load_data()
    if df is not None:
        df = clean_data(df)
        df = perform_sentiment_analysis(df)
        exploratory_analysis(df)
        build_rating_predictor(df)

        df.to_csv('Processed_Zomato_Data.csv', index=False)
        print("\n✅ Project Complete! Check output files.")

if __name__ == "__main__":
    main()
