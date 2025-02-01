# import pandas as pd
# import re
# from datetime import datetime

# def clean_text(text):
#     text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
#     text = re.sub(r"@\w+", "", text)  # Remove mentions
#     text = re.sub(r"#\w+", "", text)  # Remove hashtags
#     return text.lower()

# def preprocess_data(file_path):
#     df = pd.read_csv(file_path)
#     df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
#     df.dropna(inplace=True)
#     df["Cleaned Tweet"] = df["Tweet"].apply(clean_text)
#     df.to_csv("data/processed_data.csv", index=False)
#     print("✅ Data preprocessing complete.")

# if __name__ == "__main__":
#     preprocess_data("data/bot_detection_data.csv")

import pandas as pd
import re
import numpy as np
from datetime import datetime
from hashlib import sha256
from sklearn.ensemble import IsolationForest
import dask.dataframe as dd  # For large-scale data processing

# ⚡ Anonymization function
def anonymize_text(text):
    return sha256(str(text).encode()).hexdigest() if pd.notna(text) else text

# ⚡ Text Cleaning Function
def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters
    return text.lower().strip()

# ⚡ Feature Extraction
def extract_features(df):
    df["Cleaned Tweet"] = df["Tweet"].apply(clean_text)
    df["Anonymized User ID"] = df["User ID"].apply(anonymize_text)
    df["Anonymized Username"] = df["Username"].apply(anonymize_text)

    # Convert timestamps
    df["Created At"] = pd.to_datetime(df["Created At"], errors="coerce")
    df["Hour"] = df["Created At"].dt.hour
    df["Day"] = df["Created At"].dt.dayofweek
    df["Tweet Length"] = df["Tweet"].apply(lambda x: len(str(x)))

    # Behavioral Metrics
    df["Follower-to-Retweet Ratio"] = df["Follower Count"] / (df["Retweet Count"] + 1)
    df["Mention Ratio"] = df["Mention Count"] / (df["Tweet Length"] + 1)

    return df

# ⚡ Train Isolation Forest for Anomaly Detection
def detect_anomalies(df):
    features = ["Hour", "Day", "Tweet Length", "Follower-to-Retweet Ratio", "Mention Ratio"]
    df.dropna(subset=features, inplace=True)
    X_behavior = df[features].values

    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df["Anomaly Score"] = iso_forest.fit_predict(X_behavior)

    return df

# ⚡ Preprocessing Function
def preprocess_data(file_path):
    df = dd.read_csv(file_path)  # Use Dask for large files
    df = df.compute()  # Convert back to Pandas

    df = extract_features(df)
    df = detect_anomalies(df)

    df.drop(columns=["User ID", "Username", "Location", "Hashtags"], inplace=True)  # Remove sensitive info

    df.to_csv("data/processed_data.csv", index=False)
    print("✅ Data preprocessing complete.")

if __name__ == "__main__":
    preprocess_data("data/bot_detection_data.csv")
