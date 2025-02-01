import pandas as pd
import re
from datetime import datetime

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    return text.lower()

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
    df.dropna(inplace=True)
    df["Cleaned Tweet"] = df["Tweet"].apply(clean_text)
    df.to_csv("data/processed_data.csv", index=False)
    print("✅ Data preprocessing complete.")

if __name__ == "__main__":
    preprocess_data("data/bot_detection_data.csv")

