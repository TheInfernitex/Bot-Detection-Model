import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ⚡ Load Dataset
df = pd.read_csv("data/processed_data.csv")

# ⚡ Feature Engineering for Time-Series
df["Created At"] = pd.to_datetime(df["Created At"])
df.sort_values("Created At", inplace=True)
df.set_index("Created At", inplace=True)

# ⚡ Select Behavioral Features
features = ["Hour", "Day", "Tweet Length", "Follower-to-Retweet Ratio", "Mention Ratio"]
X = df[features].values
y = df["Bot Label"].values

# ⚡ Reshape for LSTM (samples, time steps, features)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# ⚡ Train/Test Split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ⚡ Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# ⚡ Train Model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# ⚡ Save Model
model.save("models/lstm_bot_detector")
print("✅ LSTM Model training complete.")
