import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv("../data/processed_data.csv")

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocessing: Tokenize and encode tweets
def encode_texts(texts, tokenizer, max_length=128):
    encoding = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='tf')
    return encoding['input_ids'], encoding['attention_mask']

# Encode the dataset
X, attention_mask = encode_texts(df['Cleaned Tweet'], tokenizer)
y = df['Bot Label']

# Convert TensorFlow tensors to NumPy arrays
X = X.numpy()  # Convert TensorFlow tensor to NumPy array
attention_mask = attention_mask.numpy()  # Convert TensorFlow tensor to NumPy array
y = y.to_numpy()  # Convert Pandas Series to NumPy array

# Split the dataset
X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
    X, y, attention_mask, test_size=0.2, random_state=42
)

# Load the BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Train the model
model.fit(
    [X_train, mask_train], y_train,
    validation_data=([X_test, mask_test], y_test),
    batch_size=8,
    epochs=3
)

# Save the model
model.save("models/bert_bot_detector")

print("Model training complete. Saved in 'models/' directory.")

