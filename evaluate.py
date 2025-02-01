import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("data/processed_data.csv")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
clf = joblib.load("models/bot_detector.pkl")

X_test = vectorizer.transform(df["Cleaned Tweet"])
y_test = df["Bot Label"]

y_pred = clf.predict(X_test)
print("📊 Model Performance:\n", classification_report(y_test, y_pred))

