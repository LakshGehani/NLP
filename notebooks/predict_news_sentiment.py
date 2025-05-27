import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
import torch.nn.functional as F
import pickle
from src.model import LSTMSentimentClassifier
from src.preprocess import clean_text, encode_tokens, pad_sequences

# --- Load saved model and vocab ---
with open("models/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = LSTMSentimentClassifier(vocab_size=len(vocab))
model.load_state_dict(torch.load("models/lstm_sentiment.pt"))
model.eval()

# --- Load and clean raw news ---
df = pd.read_csv("data/raw_news_AAPL_2025-05-27.csv")
texts = df['title'].astype(str).tolist()
tokens = [clean_text(text) for text in texts]

# --- Encode and pad ---
encoded = [encode_tokens(tok, vocab) for tok in tokens]
# use same length as during training
padded = pad_sequences(encoded, max_length=30)
X_tensor = torch.tensor(padded)

# --- Predict ---
with torch.no_grad():
    outputs = model(X_tensor)
    probs = F.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)

# --- Map to labels ---
label_map = {0: "positive", 1: "neutral", 2: "negative"}
df['sentiment'] = preds.numpy()
df['sentiment_label'] = df['sentiment'].map(label_map)

# --- Save to new file ---
df.to_csv("data/sentiment_news_AAPL_2025-05-27.csv", index=False)
print("Predictions saved to: data/sentiment_news_AAPL_2025-05-27.csv")
