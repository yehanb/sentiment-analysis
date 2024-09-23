import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def preprocess_tweet(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = text.lower()                  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)     # Remove special characters
    return text

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='Windows-1252')  # Specify the encoding
    except UnicodeDecodeError as e:
        print(f"Error reading the file: {e}")
        return None
    
    df['cleaned_tweet'] = df['text'].apply(preprocess_tweet)
    df['sentiment'] = df['cleaned_tweet'].apply(analyze_sentiment)
    df.dropna(subset=['cleaned_tweet'], inplace=True)
    return df

if __name__ == "__main__":
    file_path = 'data/tweets.csv'  # Path to your raw dataset
    df = load_and_preprocess_data(file_path)
    if df is not None:
        df.to_csv('data/cleaned_tweets_with_sentiment.csv', index=False)  # Save the cleaned data with sentiment
        print("Preprocessing complete. Cleaned data saved to 'data/cleaned_tweets_with_sentiment.csv'.")
    else:
        print("Preprocessing failed due to file reading errors.")
