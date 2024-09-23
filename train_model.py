import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_model(df):
    # Check for and drop rows with missing values in 'cleaned_tweet'
    df.dropna(subset=['cleaned_tweet'], inplace=True)

    X = df['cleaned_tweet']
    y = df['sentiment']

    # Convert text data to numerical data
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(classification_report(y_test, y_pred))

    # Save the model and vectorizer
    joblib.dump(model, 'model/sentiment_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')

if __name__ == "__main__":
    # Load the cleaned data
    df = pd.read_csv('data/cleaned_tweets_with_sentiment.csv')
    train_model(df)
    print("Model training complete. Model saved to 'model/sentiment_model.pkl'.")
