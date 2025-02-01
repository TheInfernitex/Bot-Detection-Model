# import pandas as pd
# import tensorflow as tf
# import numpy as np
# from transformers import BertTokenizer, TFBertForSequenceClassification
# from sklearn.metrics import classification_report

# # Load dataset
# df = pd.read_csv("data/processed_data.csv")

# # Load tokenizer and model from saved directory
# tokenizer = BertTokenizer.from_pretrained("models/bert_bot_detector")
# model = TFBertForSequenceClassification.from_pretrained("models/bert_bot_detector")

# # Preprocessing: Tokenize and encode tweets
# def encode_texts(texts, tokenizer, max_length=64):
#     encoding = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='tf')
#     return encoding['input_ids'], encoding['attention_mask']

# # Encode the dataset
# X_test, attention_mask = encode_texts(df["Cleaned Tweet"], tokenizer)
# y_test = np.array(df["Bot Label"])  # Convert labels to NumPy array

# # Make predictions
# predictions = model.predict({"input_ids": X_test, "attention_mask": attention_mask})
# y_pred = np.argmax(predictions.logits, axis=1)  # Get the predicted labels

# # Print classification report
# print("📊 Model Performance:\n", classification_report(y_test, y_pred))

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import BertTokenizer, TFBertForSequenceClassification

# ⚡ Load Preprocessed Test Data
df = pd.read_csv("data/processed_data.csv")

# ⚡ Select Features
features = ["Hour", "Day", "Tweet Length", "Follower-to-Retweet Ratio", "Mention Ratio"]
X_behavioral = df[features].values
y_test = df["Bot Label"].values

# ⚡ Load BERT Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertForSequenceClassification.from_pretrained("models/bert_bot_detector")

# ⚡ Tokenize Text for BERT
def encode_texts(texts, tokenizer, max_length=128):
    encoding = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="tf"
    )
    return encoding["input_ids"], encoding["attention_mask"]

X_text, attention_mask = encode_texts(df["Cleaned Tweet"], tokenizer)

# ⚡ Load LSTM Behavioral Model
lstm_model = tf.keras.models.load_model("models/lstm_bot_detector")

# ⚡ Predict with BERT Model
bert_logits = bert_model.predict({"input_ids": X_text, "attention_mask": attention_mask})[0]
bert_probs = tf.nn.softmax(bert_logits, axis=1).numpy()
bert_preds = np.argmax(bert_probs, axis=1)  # Convert to binary labels

# ⚡ Predict with LSTM Model
X_behavioral = np.reshape(X_behavioral, (X_behavioral.shape[0], 1, X_behavioral.shape[1]))  # Reshape for LSTM
lstm_probs = lstm_model.predict(X_behavioral).flatten()
lstm_preds = (lstm_probs > 0.5).astype(int)  # Convert probabilities to binary labels

# ⚡ Ensemble (Weighted Average)
ensemble_probs = (0.6 * bert_probs[:, 1]) + (0.4 * lstm_probs)  # Weighted combination
ensemble_preds = (ensemble_probs > 0.5).astype(int)  # Convert probabilities to binary labels

# ⚡ Compute Metrics
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"\n📊 {model_name} Performance:")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ ROC-AUC: {roc_auc:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Bot", "Bot"], yticklabels=["Not Bot", "Bot"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# ⚡ Evaluate Models
evaluate_model(y_test, bert_preds, "BERT NLP Model")
evaluate_model(y_test, lstm_preds, "LSTM Behavioral Model")
evaluate_model(y_test, ensemble_preds, "Ensemble Model (BERT + LSTM)")

print("✅ Model evaluation complete.")
