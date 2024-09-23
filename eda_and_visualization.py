import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df):
    sns.countplot(x='sentiment', data=df)
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    file_path = 'data/cleaned_tweets_with_sentiment.csv'  # Ensure this matches the output of preprocess.py
    df = pd.read_csv(file_path, encoding='Windows-1252')  # Make sure encoding is correct

    plot_sentiment_distribution(df)
