import joblib

def predict_sentiment(new_text):
    model = joblib.load('model/sentiment_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')

    new_text_vec = vectorizer.transform([new_text])

    prediction = model.predict(new_text_vec)
    return prediction[0]

if __name__ == "__main__":
    new_text = "we are sad to hear that"  # Replace with your input
    sentiment = predict_sentiment(new_text)
    print(f"The predicted sentiment is: {sentiment}")
