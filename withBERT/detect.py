import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

# Load the trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("models/bert_bot_detector")

def detect_bot(tweet):
    """Predict if a tweet is from a bot or human."""
    # Tokenize and encode the tweet
    encoding = tokenizer([tweet], truncation=True, padding=True, max_length=128, return_tensors='tf')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Get prediction
    predictions = model.predict([input_ids, attention_mask])
    predicted_class = np.argmax(predictions.logits, axis=1)
    
    return "Bot" if predicted_class == 1 else "Human"

if _name_ == "_main_":
    while True:
        tweet = input("\nEnter a tweet (or type 'exit' to quit): ")
        if tweet.lower() == "exit":
            break
        print(f" Prediction: {detect_bot(tweet)}")
