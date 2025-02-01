import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = tf.keras.models.load_model("models/bert_bot_detector")

# Preprocessing: Tokenize and encode tweets
def encode_texts(texts, tokenizer, max_length=64):
    encoding = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='tf')
    return encoding['input_ids'].numpy(), encoding['attention_mask'].numpy()

# Encode the dataset
X_test, attention_mask = encode_texts(df["Cleaned Tweet"], tokenizer)
y_test = np.array(df["Bot Label"])  # Convert labels to NumPy array

# Make predictions
predictions = model.predict({"input_ids": X_test, "attention_mask": attention_mask})
y_pred = np.argmax(predictions.logits, axis=1)  # Get the predicted labels

# Print classification report
print("📊 Model Performance:\n", classification_report(y_test, y_pred))

