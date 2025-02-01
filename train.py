import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy optimizer to fix compatibility

# Check for GPU
print("🔍 Checking for GPU...")
gpu_available = tf.config.list_physical_devices('GPU')
if gpu_available:
    use_gpu = input("🚀 GPU detected! Do you want to use it? (y/n): ").strip().lower()
    if use_gpu == 'y':
        print("✅ Using GPU for training.")
        tf.config.experimental.set_memory_growth(gpu_available[0], True)
    else:
        print("⚠️ Using CPU instead.")
else:
    print("❌ No GPU found. Using CPU.")

# Load dataset
df = pd.read_csv("data/processed_data.csv")

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocessing: Tokenize and encode tweets
def encode_texts(texts, tokenizer, max_length=64):
    encoding = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='tf')
    return encoding['input_ids'], encoding['attention_mask']

# Encode the dataset
X, attention_mask = encode_texts(df['Cleaned Tweet'], tokenizer)
y = df['Bot Label']

# Split the dataset
X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
    X, y, attention_mask, test_size=0.2, random_state=42
)

# Load the BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Ask for number of epochs
epochs = int(input("⏳ Enter number of epochs (default: 3): ") or 3)

# Train the model
model.fit(
    {"input_ids": X_train, "attention_mask": mask_train},
    y_train,
    validation_data=({"input_ids": X_test, "attention_mask": mask_test}, y_test),
    batch_size=8,
    epochs=epochs
)

# Save the model
model.save("models/bert_bot_detector")

print(f"✅ Model training complete. Saved in 'models/bert_bot_detector' after {epochs} epochs.")

