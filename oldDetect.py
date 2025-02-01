import pandas as pd
import joblib

def detect_bot(tweet):
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    clf = joblib.load("models/bot_detector.pkl")

    processed_tweet = vectorizer.transform([tweet])
    prediction = clf.predict(processed_tweet)
    return "🤖 Bot" if prediction == 1 else "👤 Human"

# Test
tweet = "This is a test tweet! #AI"
print(f"🔍 Tweet: {tweet} -> {detect_bot(tweet)}")

