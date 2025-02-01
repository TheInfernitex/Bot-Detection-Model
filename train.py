import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, AdamWeightDecay
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocessing: Tokenize and encode tweets
def encode_texts(texts, tokenizer, max_length=128):
    encoding = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="np"  # Ensure NumPy format for TensorFlow compatibility
    )
    return encoding["input_ids"], encoding["attention_mask"]

# Encode the dataset
X, attention_mask = encode_texts(df["Cleaned Tweet"], tokenizer)
y = df["Bot Label"].values  # Ensure labels are in NumPy format

# Split the dataset
X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
    X, y, attention_mask, test_size=0.2, random_state=42
)

# Convert to TensorFlow tensors
X_train_tf = tf.convert_to_tensor(X_train)
mask_train_tf = tf.convert_to_tensor(mask_train)
X_test_tf = tf.convert_to_tensor(X_test)
mask_test_tf = tf.convert_to_tensor(mask_test)
y_train_tf = tf.convert_to_tensor(y_train)
y_test_tf = tf.convert_to_tensor(y_test)

# Load the BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Use AdamWeightDecay (from Hugging Face) instead of standard Adam
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Train the model
model.fit(
    {"input_ids": X_train_tf, "attention_mask": mask_train_tf},  # Pass as dictionary
    y_train_tf,
    validation_data=(
        {"input_ids": X_test_tf, "attention_mask": mask_test_tf}, y_test_tf
    ),
    batch_size=8,
    epochs=3
)

# Save the model
model.save_pretrained("models/bert_bot_detector")

print("✅ Model training complete. Saved in 'models/' directory.")

