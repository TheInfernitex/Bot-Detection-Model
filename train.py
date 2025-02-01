import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df = pd.read_csv("data/processed_data.csv")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Cleaned Tweet"])
y = df["Bot Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save model & vectorizer
joblib.dump(clf, "models/bot_detector.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("✅ Model training complete.")

