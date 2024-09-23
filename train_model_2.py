import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # Example of another model
from sklearn.metrics import classification_report
import joblib

def train_model(df):
    df.dropna(subset=['cleaned_tweet'], inplace=True)

    X = df['cleaned_tweet']
    y = df['sentiment']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use Tfidf and include bigrams
    X_vec = vectorizer.fit_transform(X)

    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_vec)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning example
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'model/sentiment_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')

if __name__ == "__main__":
    df = pd.read_csv('data/cleaned_tweets_with_sentiment.csv')
    train_model(df)
    print("Model training complete. Model saved to 'model/sentiment_model.pkl'.")
